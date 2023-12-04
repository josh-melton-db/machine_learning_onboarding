# Databricks notebook source
pip install -q dbldatagen git+https://github.com/TimeSynth/TimeSynth.git

# COMMAND ----------

from utils.onboarding_setup import get_config, reset_tables, iot_generator

# COMMAND ----------

config = get_config(spark)
reset_tables(spark, config, dbutils)
iot_data = iot_generator(spark, config['rows_per_run'])
iot_data.write.mode('overwrite').saveAsTable(config['bronze_table'])

# COMMAND ----------

bronze_df = spark.read.table(config['bronze_table'])
bronze_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Single Node ML, Pandas, EDA
# MAGIC ### 1. Notebook capabilities - visualization, collaboration, versioning

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. MLflow Experiments
# MAGIC #####     2.1 Model / system metrics
# MAGIC #####     2.2 Visualize metrics
# MAGIC #####     2.3 Nested runs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. MLflow Model registry

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Deep learning, distributed ML

# COMMAND ----------


