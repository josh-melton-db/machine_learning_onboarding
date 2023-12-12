# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Model Inference
# MAGIC This repository is incomplete and for outline purposes only

# COMMAND ----------

pip install -q dbldatagen git+https://github.com/TimeSynth/TimeSynth.git

# COMMAND ----------

from utils.onboarding_setup import get_config, reset_tables, iot_generator
config = get_config(spark)

# COMMAND ----------

import mlflow

# TODO: separate test data
more_data = spark.read.table(config['silver_features']).toPandas()
more_data_test = more_data[int(len(more_data)*.8):]

model_name = 'lr_iot_streaming_model' #config["model_name"]
model_uri = f'models:/{model_name}/Production'
production_model = mlflow.pyfunc.load_model(model_uri)
more_data_test['predictions'] = production_model.predict(more_data_test)
more_data_test

# COMMAND ----------

# How do we make predictions on real time streams of data? Same as before, but on a streaming dataframe
more_data_stream = spark.readStream.table(config['silver_features'])

def make_predictions(microbatch_df, batch_id):
    df_to_predict = microbatch_df.toPandas()
    df_to_predict['predictions'] = production_model.predict(df_to_predict) # we use the same model and function to make predictions!
    spark.createDataFrame(df_to_predict).write.mode('overwrite').saveAsTable(config['predictions_table'])

checkpoint_location = "dbfs/tmp/josh_melton/checkpoint123" # TODO change to config checkpoint
dbutils.fs.rm(checkpoint_location, True) # data was overwritten so we remove the checkpoint
(
  more_data_stream.writeStream
  .format("delta")
  .option("checkpointLocation", checkpoint_location)
  .option("mergeSchema", "true") # if you want the schema of the target table to automatically add new columns 
                                 # that are encountered, add this option. Can also be done in spark config
  .foreachBatch(make_predictions) 
  .outputMode("update")
  .trigger(availableNow=True) # if you want to run constantly and check for new data at some interval, use trigger(processingTime='5 minutes')
  # .queryName("example_query") # use this for discoverability in the Spark UI
  .start()
).awaitTermination()

# COMMAND ----------

# Commercial databricks also supports model, feature, and function serving APIs for real time inference
