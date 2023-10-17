# Databricks notebook source
# MAGIC %md ### Writing a Spark DataFrame to MDS (Mosaic Data Shard) and JSONL formats
# MAGIC
# MAGIC This notebook was developed on ML DBR 14.0. It downloads the training and testing [mosaicml/instruct-v3](https://huggingface.co/datasets/mosaicml/instruct-v3) datasets from the huggingface datasets hub and converts them to Spark Dataframes. Then, MosaicML's streaming library is used to [convert the dataframes](https://docs.mosaicml.com/projects/streaming/en/stable/examples/spark_dataframe_to_MDS.html) to [MDS format](https://docs.mosaicml.com/projects/streaming/en/stable/fundamentals/dataset_format.html#formats) persisted in cloud storage.  Pandas and the s3fs libary are also used to write version of the datasets in JSONL format.  
# MAGIC
# MAGIC Use of the MDS and JSONL versions of the datasets depends on which MosaicML training API is used. For the standard API (mcli run), the MDS format should be used. For the finetune API, the JSONL version of the datasets must be used; this is because the finetune API handles the conversion to MDS itself. There is currently no option to pass MDS datasets directly to the finetuning API.  
# MAGIC
# MAGIC The finetuning API is a higher-level API and automatically saves models in a huggingface transformers compatible format, making it easy to load the tuned model in Databricks and log it to MLFlow.
# MAGIC
# MAGIC The cloud storage paths where the data is stored are referenced in MosaicML's fine tuning yaml config file. The MDS files will be temporarily copied into the MosaicML compute plane during model training. Trained model artifacts will then be persisted back to a cloud storage path defined in the yaml config.
# MAGIC
# MAGIC NOTE: This example assumes [AWS Access Keys](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/s3.html#aws-s3) have been stored as [environment variables](https://docs.databricks.com/en/storage/amazon-s3.html#access-s3-buckets-with-uris-and-aws-keys) on the cluster.

# COMMAND ----------

# MAGIC %pip install 'mosaicml-streaming[databricks]>=0.6,<0.7'

# COMMAND ----------

#%pip install git+https://github.com/mosaicml/streaming.git@703afa6cda5d53fcefca45b3f0534918ea8e6247 s3fs

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from collections import namedtuple
from datasets import load_dataset
from pyspark.sql.types import StructType, StructField, StringTypec
from streaming.base.converters import dataframeToMDS

# COMMAND ----------

# DBTITLE 1,Load Clean Final Data Set
all_dataframe = spark.table("main.mosaic_end_to_end.gold_training_final_set").select("prompt", "response")

train, test = all_dataframe.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

display(all_dataframe)

# COMMAND ----------

# MAGIC %md # Persist to MDS
# MAGIC
# MAGIC ### 2 Options
# MAGIC
# MAGIC 1. Write to UC Managed Volumes to have all source data governed and managed in one place
# MAGIC 2. Write to External S3 Bucket anywhere

# COMMAND ----------

# DBTITLE 1,Convert Spark Data Frames to MDS
# Configure MDS writer
def to_db_path(path):
  return path.replace('/dbfs/', 'dbfs:/')

local_path = 'dbfs:/Users/cody.davis@databricks.com/mosaicml/fine_tuning'

path = 's3://codymosaic/datasets'
train_path = f'{path}/train'
test_path = f'{path}/test'

columns = {'prompt': 'str', 'response': 'str'}

train_kwargs = {'out': train_path, 
                'columns': columns}
          
test_kwargs = train_kwargs.copy()
test_kwargs['out'] = test_path

# Remove exist MDS files if they exist
try:
  if dbutils.fs.ls(train_path) or dbutils.fs.ls(test_path):
    dbutils.fs.rm(path, recurse=True)

except:
  dbutils.fs.mkdirs(train_path)
  dbutils.fs.mkdirs(test_path)

def write_to_mds(df, kwargs):
  dataframeToMDS(df.repartition(8), 
                merge_index=True,
                mds_kwargs=kwargs

  )


# COMMAND ----------

print(train_path)
print(test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Option 1: Persisting MDS to UC Volumes

# COMMAND ----------

# DBTITLE 1,Write to Managed or External UC Volume
# UC Volume path
train_volume_path = "/Volumes/main/mosaic_end_to_end/mosaic_source_data/train"


columns = {'prompt': 'str', 'response': 'str'}

train_kwargs = {'out': train_volume_path, 
                'columns': columns}
          

## Write Training Data Frame to Volume
dataframeToMDS(train.repartition(8), 
                merge_index=True,
                mds_kwargs=train_kwargs

  )

# COMMAND ----------

# DBTITLE 1,Write Test Data Set to UC Managed Volume
# UC Volume path
test_volume_path = "/Volumes/main/mosaic_end_to_end/mosaic_source_data/test"


columns = {'prompt': 'str', 'response': 'str'}

test_kwargs = {'out': test_volume_path, 
                'columns': columns}
          

## Write Training Data Frame to Volume
dataframeToMDS(test.repartition(8), 
                merge_index=True,
                mds_kwargs=test_kwargs

  )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Option 2: Write to External S3 Bucket

# COMMAND ----------

# DBTITLE 1,Write Training Set to External Bucket
write_to_mds(train, train_kwargs)

# COMMAND ----------

# DBTITLE 1,Write Test Set to External Bucket
write_to_mds(test, test_kwargs)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ### Write to JSONL to External Bucket

# COMMAND ----------

train_pd = train.select("prompt", "response").toPandas()
test_pd = test.select("prompt", "response").toPandas()

# COMMAND ----------

train_pd.to_json("s3://codymosaic/datasets/jsonl/train.jsonl",
                 orient="records",
                 lines=True)

test_pd.to_json("s3://codymosaic/datasets/jsonl/test.jsonl",
                 orient="records",
                 lines=True)

# COMMAND ----------

dbutils.fs.ls("s3://codymosaic/datasets/")
