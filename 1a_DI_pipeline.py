# Databricks notebook source
# MAGIC %pip install --quiet databricks-sdk==0.24.0 mlflow==2.14.1 unstructured==0.13.7 sentence-transformers==3.0.1 torch==2.3.0 transformers==4.40.1 accelerate==0.27.2
# MAGIC %pip install "unstructured[pdf]"
# MAGIC %pip install pymupdf
# MAGIC %pip install pypdf2

# COMMAND ----------

import os
import re
import pandas as pd
from pyspark.sql.functions import col, lit
from unstructured.partition.pdf import partition_pdf
from PyPDF2 import PdfReader

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

catalog_name = "kyra_wulffert"
schema_name = "poc_doc_management"
volume_name = "docs"
processed_table_name = "bronze_docs"

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

from unstructured.partition.pdf import partition_pdf
from PyPDF2 import PdfReader

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
            cleaned_first_page_text = first_page_text.strip()

            # Extract year from cleaned text
            year = extract_year_from_text(cleaned_first_page_text)

            # Fallback: Search entire document if no year is found
            if not year:
                full_text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
                year = extract_year_from_text(full_text.strip())

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


# COMMAND ----------

from pyspark.sql.functions import col, lower

# Normalize file names in bronze_docs
processed_files_df = spark.table(f"{catalog_name}.{schema_name}.bronze_docs") \
                           .select(col("file_name")) \
                           .withColumn("file_name", lower(col("file_name")))

display(processed_files_df)

# COMMAND ----------

# Walk through the volume and collect files
volume_files = []
for root, dirs, files in os.walk(volume_path):
    for file in files:
        volume_files.append({
            "file_name": file.lower(),  # Normalize for matching
            "full_path": os.path.join(root, file)  # Include full path
        })

# Convert volume files to Spark DataFrame
volume_files_df = spark.createDataFrame(volume_files)

# Identify new files by excluding already processed ones
new_files_df = volume_files_df.join(
    processed_files_df, on="file_name", how="leftanti"
)

# Collect new files with their full paths
new_files = [(row["file_name"], row["full_path"]) for row in new_files_df.collect()]

# Debug: Print the files to process
print("Final list of files to process with paths:")
for file_name, file_path in new_files:
    print(f"File Name: {file_name}, Full Path: {file_path}")

# Process new files
data = []
for file_name, file_path in new_files:
    print(f"Processing file: {file_name}")
    file_data = process_file(file_path)  # Use the full path for processing
    data.append(file_data)


# COMMAND ----------

# Test output by converting processed data to Pandas DataFrame
if data:
    new_files_df = pd.DataFrame(data)
    print(f"Processed {len(data)} new files. Here's the output:")
    print(new_files_df.head())  # Display a sample of the processed data
else:
    print("No new files to process.")

# COMMAND ----------

# Convert processed data to a Pandas DataFrame
if data:
    new_files_df = pd.DataFrame(data)

    # Append new data to the bronze_docs table
    new_files_spark_df = spark.createDataFrame(new_files_df)
    new_files_spark_df.write.format("delta").mode("append").saveAsTable(f"{catalog_name}.{schema_name}.bronze_docs")

    print(f"Processed and appended {len(data)} new files.")
else:
    print("No new files to process.")