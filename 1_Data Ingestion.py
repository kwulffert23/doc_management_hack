# Databricks notebook source
# MAGIC %pip install --quiet databricks-sdk==0.24.0 mlflow==2.14.1 unstructured==0.13.7 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 accelerate==0.27.2
# MAGIC %pip install "unstructured[pdf]"
# MAGIC %pip install pymupdf
# MAGIC %pip install pypdf2

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

# MAGIC %md
# MAGIC ## Data Ingestion

# COMMAND ----------

catalog_name = "kyra_wulffert"
schema_name = "poc_doc_management"
volume_name = "docs"

# COMMAND ----------

def extract_year_from_text(text):
    """
    Extracts the publication year from text using various patterns.
    Looks for:
    - "Publication Date: YYYY"
    - "in <Month> <Year>"
    - Standalone dates like "30 November 2023" or "November 2023"
    """
    # Pattern for "Publication Date: YYYY"
    match = re.search(r"Publication date:.*?(\d{4})", text, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern for "in <Month> <Year>" or similar phrases
    match = re.search(
        r"(published|in|updated).*?\b(January|February|March|April|May|June|July|August|September|October|November|December)\b.*?(\d{4})",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(3)  # Return the year

    # Pattern for standalone dates
    match = re.search(
        r"(\b\d{1,2}\b\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})|((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})",
        text,
        re.IGNORECASE,
    )
    if match:
        # Extract the year from standalone date
        date = match.group(0)
        year_match = re.search(r"\d{4}", date)
        if year_match:
            return year_match.group(0)

    return None



# COMMAND ----------

# Clean extracted text
def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary line breaks
    and collapsing multiple spaces into single spaces.
    """
    if text:
        # Remove line breaks and collapse multiple spaces
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
    return text


# COMMAND ----------

# Define your catalog location
volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

# Initialize a list to store data
data = []

# Function to process a single file
def process_file(file_path):
    """
    Processes a file, extracts text and metadata.
    """
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1].lower()

    try:
        if file_extension == ".pdf":
            # Extract text and metadata
            pdf_elements = partition_pdf(file_path)
            text = "\n".join([element.text for element in pdf_elements if element.text])

            # Extract the first page's text
            pdf_reader = PdfReader(file_path)
            first_page_text = pdf_reader.pages[0].extract_text()
            
            # Clean the extracted first page text
            cleaned_first_page_text = clean_text(first_page_text)
            # print("Cleaned First Page Text:", cleaned_first_page_text)  # Debugging

            # Extract year from cleaned text
            year = extract_year_from_text(cleaned_first_page_text)

            # Fallback: Search entire document if no year is found
            if not year:
                full_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                cleaned_full_text = clean_text(full_text)
                year = extract_year_from_text(cleaned_full_text)

            num_pages = len(pdf_reader.pages)

        else:
            text = f"Unsupported file type: {file_extension}"
            year = None
            num_pages = None

        length = len(text) if text else 0

        return {
            "file_name": file_name,
            "file_extension": file_extension,
            "num_pages": num_pages,
            "length": length,
            "year": year,
            "text": text,
        }

    except Exception as e:
        return {
            "file_name": file_name,
            "file_extension": file_extension,
            "num_pages": None,
            "length": 0,
            "year": None,
            "text": f"Error processing file: {str(e)}",
        }

# Walk through the directory and process files
for root, dirs, files in os.walk(volume_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_data = process_file(file_path)
        data.append(file_data)

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# COMMAND ----------

df

# COMMAND ----------

spark.createDataFrame(df) \
    .write.format("delta") \
    .option("mergeSchema", "true") \
    .mode("overwrite") \
    .saveAsTable(f"{catalog_name}.{schema_name}.bronze_docs")