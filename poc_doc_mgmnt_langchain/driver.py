# Databricks notebook source
# MAGIC %md
# MAGIC # Driver notebook
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export. We generated three notebooks in the same folder:
# MAGIC - [agent]($./agent): contains the code to build the agent.
# MAGIC - [config.yml]($./config.yml): contains the configurations.
# MAGIC - [**driver**]($./driver): logs, evaluate, registers, and deploys the agent.
# MAGIC
# MAGIC This notebook uses Mosaic AI Agent Framework ([AWS](https://docs.databricks.com/en/generative-ai/retrieval-augmented-generation.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/retrieval-augmented-generation)) to deploy the agent defined in the [agent]($./agent) notebook. The notebook does the following:
# MAGIC 1. Logs the agent to MLflow
# MAGIC 2. Evaluate the agent with Agent Evaluation
# MAGIC 3. Registers the agent to Unity Catalog
# MAGIC 4. Deploys the agent to a Model Serving endpoint
# MAGIC
# MAGIC ## Prerequisities
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.
# MAGIC - Review the contents of [config.yml]($./config.yml) as it defines the tools available to your agent, the LLM endpoint, and the agent prompt.
# MAGIC - Review and run the [agent]($./agent) notebook in this folder to view the agent's code, iterate on the code, and test outputs.
# MAGIC
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See docs ([AWS](https://docs.databricks.com/en/generative-ai/deploy-agent.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/generative-ai/deploy-agent)) for details

# COMMAND ----------

# MAGIC %pip install -U -qqqq databricks-agents mlflow langchain==0.2.16 langgraph-checkpoint==1.0.12  langchain_core langchain-community==0.2.16 langgraph==0.2.16 pydantic langchain_databricks
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog_name = "kyra_wulffert"
schema_name = "poc_doc_management"
volume_name = "docs"
raw_table_name = "bronze_docs"
VS_INDEX = "db_doc_poc_doc_management_sequential"


# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Log the agent as code from the [agent]($./agent) notebook. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Log the model to MLflow
import os
import mlflow

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "what is dso?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.langchain.log_model(
        lc_model=os.path.join(
            os.getcwd(),
            'agent',
        ),
        pip_requirements=[
            "langchain==0.2.16",
            "langchain-community==0.2.16",
            "langgraph-checkpoint==1.0.12",
            "langgraph==0.2.16",
            "pydantic",
            "langchain_databricks", # used for the retriever tool
        ],
        model_config="config.yml",
        artifact_path='agent',
        input_example=input_example,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate synthetic data

# COMMAND ----------

docs_df = (
    spark.table(f"{catalog_name}.{schema_name}.{raw_table_name}")
    .withColumnRenamed("text", "content")
    .withColumnRenamed("file_name", "doc_uri")
    
)
display(docs_df)

# COMMAND ----------

# Optional guideline
guidelines = """
You are a generator tasked with creating an evaluation dataset for a Q&A chatbot that uses a Retrieval-Augmented Generation (RAG) solution. The chatbot is designed to provide accurate, concise, and relevant answers based on given documents in the knowledge base of the RAG solution.

### Task Description:
Your job is to generate realistic, varied, and challenging question-answer pairs that will be used to test the chatbot's performance. Each question must be crafted with the following guidelines:
1. Questions should reflect the information needs of professionals working at energy supplier companies, including operations managers, regulatory affairs specialists, and technical engineers.
2. Answers should be relevant to the given documents and not include hallucinated or invented information.
3. Include a clear reference to the source document or section from which the answer is derived (e.g., "Source: RIIO-ED2 Final Determinations Document, Chapter 3, Page 12").

### Content Guidance:
1. Questions should span the following categories:
   - Regulatory policies and compliance.
   - Financial incentives and penalties.
   - Distribution network operations and infrastructure.
   - Consumer impact assessments.
   - Sustainability and net-zero goals.

2. Ensure the dataset contains a mix of:
   - Direct factual questions (e.g., "What are the financial incentives for exceeding sustainability goals?").
   - Interpretive questions (e.g., "How does the RIIO-ED2 framework address consumer benefits?").
   - Context-specific scenarios (e.g., "As an energy supplier, how should we respond to changes in incentive structures outlined in the RIIO-ED2?").

### Style Guidelines:
1. Answers should be concise but detailed enough to be actionable for professionals.
2. Use formal and professional language.
3. Avoid jargon unless necessary, and define technical terms when used.

### Personas:
1. **Regulatory Affairs Specialist**:
   - Focus on compliance requirements, policies, and implications of determinations.
   - Example question: "What penalties are outlined for non-compliance with the RIIO-ED2 determinations?"

2. **Operations Manager**:
   - Interested in operational impact and infrastructure-related incentives.
   - Example question: "What incentives are available for modernizing distribution networks?"

3. **Technical Engineer**:
   - Seeks technical details on energy infrastructure and implementation.
   - Example question: "What are the technical specifications for achieving net-zero goals in distribution networks?"

4. **Consumer Advocacy Lead**:
   - Concerned with consumer impact and benefits.
   - Example question: "How does the RIIO-ED2 framework ensure consumer affordability?"

### Example Question-Answer Pairs:
1. **Question**: "What are the financial incentives for DNOs to improve stakeholder engagement?"
   **Answer**: "The RIIO-ED2 framework includes stakeholder satisfaction surveys as part of the performance assessment, rewarding DNOs with financial incentives for high engagement scores. Source: RIIO-ED2 Final Determinations Document, Chapter 4, Page 18."

2. **Question**: "What are the penalties for failing to meet net-zero targets?"
   **Answer**: "DNOs failing to meet net-zero targets may face reduced allowances and potential regulatory action. Source: RIIO-ED2 Final Determinations Document, Chapter 7, Page 42."

3. **Question**: "What are the requirements for DNOs under the RIIO-ED2 to improve infrastructure resilience?"
   **Answer**: "DNOs are required to implement advanced distribution management systems and reinforce network infrastructure to ensure resilience. Source: RIIO-ED2 Final Determinations Document, Chapter 5, Page 30."

Generate a diverse set of such question-answer pairs, ensuring they align with the personas, categories, and guidelines provided. Include the reference for each answer and ensure factual alignment with the source material.
"""

# COMMAND ----------

from pyspark.sql.functions import col, to_json, struct, expr, lit
from databricks.agents.evals import generate_evals_df

# Convert the list to a Spark DataFrame
docs_sample_df = spark.createDataFrame(docs_df.head(20))

# Generate 1 question for each document
synthetic_eval_data = generate_evals_df(
    docs=docs_sample_df,
    guidelines=guidelines, 
    num_evals=20
)

display(synthetic_eval_data)

# COMMAND ----------

import mlflow
import pandas as pd

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",  # replace `chain` with artifact_path that you used when calling log_model.
        data=synthetic_eval_data,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "kyra_wulffert"
schema = "poc_doc_management"
model_name = "doc_management_agent"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents

# Deploy the model to the review app and a model serving endpoint
agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"})