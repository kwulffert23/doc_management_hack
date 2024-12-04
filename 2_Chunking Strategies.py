# Databricks notebook source
# MAGIC %pip install --quiet databricks-sdk==0.24.0 mlflow==2.14.1 unstructured==0.13.7 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 accelerate==0.27.2
# MAGIC %pip install "unstructured[pdf]"
# MAGIC %pip install pymupdf
# MAGIC %pip install pypdf2
# MAGIC %pip install databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


import json, sentence_transformers, torch, yaml, os, gc, logging, time, requests, mlflow
import matplotlib.pyplot as plt
import pandas as pd
import pyspark.sql.functions as F

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import EndpointType, VectorIndexType, DeltaSyncVectorIndexSpecResponse, EmbeddingSourceColumn, PipelineType
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, ChatMessage, ChatMessageRole
from langchain_core.embeddings import Embeddings
from mlflow.tracking.client import MlflowClient
from pyspark.sql.types import StringType, StructField, StructType, ArrayType, IntegerType
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.text import partition_text
import tempfile
from typing import List, Callable

# COMMAND ----------

import os
import re
import pandas as pd
from unstructured.partition.pdf import partition_pdf
from PyPDF2 import PdfReader

# COMMAND ----------

catalog_name = "kyra_wulffert"
schema_name = "poc_doc_management"
volume_name = "docs"

# COMMAND ----------

# MAGIC %md
# MAGIC # Chunk the data

# COMMAND ----------

# MAGIC %md
# MAGIC ## RecursiveCharacterTextSplitter

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

@F.udf(ArrayType(StructType([
  StructField("content", StringType(), True),
  StructField("char_length", IntegerType(), True),
  StructField("chunk_num", IntegerType(), True)
])))


def recursive_splitter_udf(doc_text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Splits a document into chunks using RecursiveCharacterTextSplitter.

    Args:
        doc_text (str): The input document as a single string.
        chunk_size (int): Maximum number of characters in each chunk.
        chunk_overlap (int): Number of overlapping characters between consecutive chunks.

    Returns:
        list: A list of dictionaries containing chunk content and metadata.
    """
    if not doc_text:
        return []

    # Initialize the text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split the text
    chunks = splitter.split_text(doc_text)
    
    # Return chunks with metadata
    return [
        {"content": chunk, "char_length": len(chunk), "chunk_num": i}
        for i, chunk in enumerate(chunks)
    ]

# COMMAND ----------

doc_chunks_recursive = (
    spark
    .table(f"{catalog_name}.{schema_name}.bronze_docs")
    .withColumn(
        "chunks",
        recursive_splitter_udf(
            F.col("text"),
            F.lit(1500),
            F.lit(500)
        )
    )
    # Explode the chunks while retaining all original columns
    .withColumn("chunk", F.explode("chunks"))
    # Retain all original fields and include chunk fields
    .select("*", F.col("chunk.*"))
    # Add a unique UUID based on file_name and chunk_num
    .withColumn("uuid", F.concat_ws("_", F.col("file_name"), F.col("chunk_num")))
    .drop("text","chunk", "chunks")  
)


# COMMAND ----------

display(doc_chunks_recursive.limit(10))

# COMMAND ----------

df_recursive_pd = doc_chunks_recursive.toPandas()

# COMMAND ----------

df_recursive_pd.describe()

# COMMAND ----------

# Now we save the gold table and setup a vector search endpoint 
(
  doc_chunks_recursive
  .write
  .option("delta.enableChangeDataFeed", "true") # enable CDF
  .option("mergeSchema", "true") 
  .mode("overwrite")
  .saveAsTable(f"{catalog_name}.{schema_name}.gold_doc_chunks_recursive") 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sequential chunking with size constraints

# COMMAND ----------

def combine_partitioned_elements(elements, max_characters):
    combined_elements = []
    current_element = ""
    
    for element in elements:
        element_text = element.text.strip()
        if len(current_element) + len(element_text) <= max_characters:
            current_element += " " + element_text
        else:
            combined_elements.append(current_element.strip())
            current_element = element_text
    
    if current_element:
        combined_elements.append(current_element.strip())
    
    return combined_elements

# COMMAND ----------

def combine_small_chunks(chunks, max_characters):
    merged_chunks = []
    current_chunk = {"content": "", "char_length": 0}

    for chunk in chunks:
        if current_chunk["char_length"] + len(chunk.text) <= max_characters:
            current_chunk["content"] += " " + chunk.text
            current_chunk["char_length"] += len(chunk.text)
        else:
            merged_chunks.append(current_chunk)
            current_chunk = {"content": chunk.text, "char_length": len(chunk.text)}

    if current_chunk["content"]:
        merged_chunks.append(current_chunk)
    
    return merged_chunks

# COMMAND ----------

from unstructured.partition.text import partition_text
from unstructured.chunking.basic import chunk_elements

@F.udf(ArrayType(StructType([
    StructField("content", StringType(), True),
    StructField("category", StringType(), True),
    StructField("char_length", IntegerType(), True),
    StructField("chunk_num", IntegerType(), True)
])))
def get_doc_chunks(doc_text: str, max_characters: int, new_after_n_chars: int) -> list:
    """
    Splits the document text into chunks using unstructured's chunking and partitioning tools.

    Args:
        doc_text (str): The input document text.
        max_characters (int): Maximum number of characters per chunk.
        new_after_n_chars (int): Threshold to start a new chunk.

    Returns:
        list: List of dictionaries containing chunk details.
    """
    # Ensure partition_text produces Element objects
    elements = partition_text(text=doc_text)

    # Check that elements are properly formatted
    if not all(hasattr(el, 'text') for el in elements):
        raise ValueError("Partitioned elements do not have the required 'text' attribute.")

    # Use chunk_elements with proper input
    chunks = chunk_elements(elements, max_characters=max_characters, new_after_n_chars=new_after_n_chars)

    # Prepare the results with metadata
    ret = []
    for i, _chunk in enumerate(chunks):
        _cat = getattr(_chunk, "category", "Unknown")
        _content = _chunk.metadata.text_as_html if (_cat == "Table") else _chunk.text  # Keep HTML for tables
        ret.append({
            "content": _content,
            "category": _cat,
            "char_length": len(_content),
            "chunk_num": i
        })

    return ret


# COMMAND ----------

doc_chunks_sequential = (
  spark
  .table(f"{catalog_name}.{schema_name}.bronze_docs")
  .withColumn(
    "chunks",
    get_doc_chunks(F.column("text"), F.lit(1500), F.lit(1000))
  )
  # Explode the chunks while retaining all original columns
  .withColumn("chunk", F.explode("chunks"))
  # Retain all original fields and include chunk fields
  .select("*", F.col("chunk.*"))
  # Add a unique UUID based on file_name and chunk_num
  .withColumn("uuid", F.concat_ws("_", F.col("file_name"), F.col("chunk_num")))
  .drop("text","chunk", "chunks")  
)

# COMMAND ----------

display(doc_chunks_sequential.limit(10))

# COMMAND ----------

df_sequential = doc_chunks_sequential.toPandas()

# COMMAND ----------

df_sequential.describe()

# COMMAND ----------

# Now we save the gold table and setup a vector search endpoint 
(
  doc_chunks_sequential
  .write
  .option("delta.enableChangeDataFeed", "true") # enable CDF
  .option("mergeSchema", "true") 
  .mode("overwrite")
  .saveAsTable(f"{catalog_name}.{schema_name}.gold_doc_chunks_sequential") 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agentic Chunking

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate

LLAMA3_ENDPOINT = "databricks-meta-llama-3-1-70b-instruct"
llm = ChatDatabricks(endpoint=LLAMA3_ENDPOINT)

def agentic_chunking(text_data):

    prompt = """
      I am providing a document below.

      Split the document into chunks that maintain semantic coherence, ensuring each chunk represents a complete and meaningful unit of information. 
      Each chunk should include the original text from the document, without any modifications or paraphrasing. 
      Use your understanding of the document’s structure, topics, and flow to identify natural breakpoints, such as paragraphs or sections. 
      Do not impose any character or word limit on the chunks; instead, focus on preserving the logical flow and context.

      Return the response strictly in the following JSON-like format, with no additional text or explanation:

      [
          {{
              "chunk_number": 1,
              "content": "Content of the first chunk."
          }},
          {{
              "chunk_number": 2,
              "content": "Content of the second chunk."
          }},
          ...
      ]

      Document:
      {document}
      """
    prompt_template = PromptTemplate.from_template(prompt)
    chain = prompt_template | llm
    result = chain.invoke({"document": text_data})
    return result

# COMMAND ----------

llm_response = agentic_chunking(df['text'][0])

# COMMAND ----------

llm_response.content

# COMMAND ----------

def parse_llm_output_to_dataframe(llm_response):
    """
    Parses the JSON-like string response from LLM and converts it to a Pandas DataFrame.
    
    Args:
        llm_response (str): The JSON-like string response from the LLM.

    Returns:
        pd.DataFrame: A DataFrame with chunk_number and content columns.
    """
    # Strip single quotes around the JSON-like string and parse it
    try:
        parsed_data = json.loads(llm_response.strip("'"))
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

    # Convert to DataFrame
    df = pd.DataFrame(parsed_data)
    return df

# COMMAND ----------

chunk_table = parse_llm_output_to_dataframe(llm_response.content)

# COMMAND ----------

chunk_table.describe()

# COMMAND ----------

chunk_table.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create the Vector Search Index

# COMMAND ----------

# VS_ENDPOINT = f"vs_endpoint_{(sum(ord(char) for char in DA.unique_name('_')) % 9) + 1}"
VS_ENDPOINT = "one-env-shared-endpoint-16"

w = WorkspaceClient()

try:
    endpoint = w.vector_search_endpoints.get_endpoint(VS_ENDPOINT)
    print(f"Endpoint {VS_ENDPOINT} found. Using this endpoint for your index.")
except:
    print(f"Endpoint {VS_ENDPOINT} not found. Please confirm the endpoint has been set up.")

# COMMAND ----------

VS_INDEX = f"db_doc_{schema_name}_sequential" 

# COMMAND ----------

try:
    w.vector_search_indexes.sync_index(f"{catalog_name}.{schema_name}.{VS_INDEX}")
except ResourceDoesNotExist as e:
    w.vector_search_indexes.create_index(
        name=f"{catalog_name}.{schema_name}.{VS_INDEX}",
        endpoint_name=VS_ENDPOINT,
        primary_key="uuid",
        index_type=VectorIndexType("DELTA_SYNC"),
        delta_sync_index_spec=DeltaSyncVectorIndexSpecResponse(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    name="content",
                    embedding_model_endpoint_name="databricks-gte-large-en"
                )],
            pipeline_type=PipelineType("TRIGGERED"),
            source_table=f"{catalog_name}.{schema_name}.gold_doc_chunks_sequential"
        )
    )


# COMMAND ----------

status = w.vector_search_indexes.get_index(f"{catalog_name}.{schema_name}.{VS_INDEX}").as_dict()

print(status.get("status")["ready"])

status

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building The Retriever

# COMMAND ----------

def get_relevant_documents(question:str, index_name:str, k:int = 3, filters:str = None, max_retries:int = 3) -> List[dict]:
    """
    This function searches through the supplied vector index name and returns relevant documents 
    """
    docs = w.vector_search_indexes.query_index(
        index_name=index_name,
        columns=["uuid", "content", "char_length", "chunk_num", "file_name", "file_extension", "num_pages", "length", "year"],
        filters_json=filters,
        num_results=k,
        query_text=question
    )
    docs_pd = pd.DataFrame(docs.result.data_array)
    docs_pd.columns = [_c.name for _c in docs.manifest.columns]
    return json.loads(docs_pd.to_json(orient="records"))

# COMMAND ----------

# NOTE: This can take up to 7min to be ready
while w.vector_search_indexes.get_index(f"{catalog_name}.{schema_name}.{VS_INDEX}").status.ready is not True:
    print("Vector search index is not ready yet...")
    time.sleep(30)

print("Vector search index is ready")

# COMMAND ----------

VS_INDEX_FULL_NAME=f"{catalog_name}.{schema_name}.{VS_INDEX}"
get_relevant_documents("What is Distribution System Operation?", VS_INDEX_FULL_NAME, k=10)

# COMMAND ----------

import os
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
index = vsc.get_index(endpoint_name="one-env-shared-endpoint-16", index_name="kyra_wulffert.poc_doc_management.db_doc_poc_doc_management_sequential")

results = index.similarity_search(
    columns=["file_name", "year", "content"],
    query_text="What are the key performance metrics that Distribution Network Operators (DNOs) must achieve under the RIIO-ED2 framework to receive financial incentives?",
    query_type="hybrid"
)
display(results)