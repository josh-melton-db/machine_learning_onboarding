# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Model Inference
# MAGIC This repository is incomplete and for outline purposes only

# COMMAND ----------

pip install -q dbldatagen

# COMMAND ----------

from utils.onboarding_setup import get_config, reset_tables, iot_generator
config = get_config(spark)

# COMMAND ----------

import mlflow

feature_data = spark.read.table(config['silver_features']).toPandas()

model_uri = f'models:/{config["model_name"]}/Production'
production_model = mlflow.pyfunc.load_model(model_uri)
feature_data['predictions'] = production_model.predict(feature_data)
feature_data

# COMMAND ----------

# MAGIC %md
# MAGIC How do we make predictions on real time streams of data? Same as before, but on a streaming dataframe

# COMMAND ----------

# DBTITLE 1,Streaming Inference

more_data_stream = spark.readStream.table(config['silver_features'])

def make_predictions(microbatch_df, batch_id):
    df_to_predict = microbatch_df.toPandas()
    df_to_predict['predictions'] = production_model.predict(df_to_predict) # we use the same model and function to make predictions!
    spark.createDataFrame(df_to_predict).write.mode('overwrite').saveAsTable(config['predictions_table'])

dbutils.fs.rm(config['checkpoint_location'], True) # source data was overwritten during setup so we remove any existing checkpoints

# COMMAND ----------

(
  more_data_stream.writeStream
  .format("delta")
  .option("checkpointLocation", config['checkpoint_location'])
  .option("mergeSchema", "true") # if you want the schema of the target table to automatically add new columns 
                                 # that are encountered, add this option. Can also be done in spark config
  .foreachBatch(make_predictions) 
  .outputMode("update")
  .trigger(availableNow=True) # if you want to run constantly and check for new data at some interval, use trigger(processingTime='5 minutes')
  # .queryName("example_query") # use this for discoverability in the Spark UI
  .start()
).awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC Commercial databricks also supports model, feature, and function serving APIs for real time inference
