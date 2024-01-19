# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction

# COMMAND ----------

# DBTITLE 1,Import Config
from utils.onboarding_setup import get_config
config = get_config(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Batch Inference

# COMMAND ----------

# DBTITLE 1,Make Predictions
import mlflow

feature_data = spark.read.table(config['silver_features']).toPandas()

model_uri = f'models:/{config["model_name"]}/Production'
production_model = mlflow.pyfunc.load_model(model_uri)
feature_data['predictions'] = production_model.predict(feature_data)
feature_data

# COMMAND ----------

# MAGIC %md
# MAGIC You can also load MLflow models to make predictions in parallel using Spark, as [demonstrated here](https://docs.databricks.com/en/mlflow/model-example.html) and on the artifacts tab of your MLflow UI

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Streaming Inference

# COMMAND ----------

# MAGIC %md
# MAGIC How do we make predictions on real time streams of data? Same as before, but on a streaming dataframe. We'll put our logic in a function this time. The foreachBatch calls the function on each microbatch in our streaming dataframe

# COMMAND ----------

# DBTITLE 1,Define Function

feature_data_stream = spark.readStream.table(config['silver_features'])

def make_predictions(microbatch_df, batch_id):
    df_to_predict = microbatch_df.toPandas()
    df_to_predict['predictions'] = production_model.predict(df_to_predict) # we use the same model and function to make predictions!
    spark.createDataFrame(df_to_predict).write.mode('overwrite').option('mergeSchema', 'true').saveAsTable(config['predictions_table'])

# COMMAND ----------

# DBTITLE 1,Run Streaming Inference
dbutils.fs.rm(config['checkpoint_location'], True) # source data was overwritten during setup so we remove any existing checkpoints
(
  feature_data_stream.writeStream
  .format("delta")
  .option("checkpointLocation", config['checkpoint_location'])
  .foreachBatch(make_predictions) # run our prediction function on each microbatch
  .trigger(availableNow=True) # if you want to run constantly and constantly check for new data, comment out this line
  .queryName(f'stream_to_{config["predictions_table"]}') # use this for discoverability in the Spark UI
  .start()
).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC Commercial databricks instances also support real time model, feature, and function serving APIs. Check out our documentation for more information on those topics
