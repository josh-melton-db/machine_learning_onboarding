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

# 1. Notebook capabilities - visualization, collaboration, versioning
# 2. MLflow Experiments
#     1. Model / system metrics
#     2. Visualize metrics
# 3. MLflow Model registry
# 4. Deep learning (distributed ML)
# 5. Gotchas 
#     1. Metrics must be numbers (not arbitrary text)
#     2. Nesting limitations
#     3. Managing state of a run, accessing a specific run afterwards
