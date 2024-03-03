# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

from onboarding_setup import generate_iot, dgconfig

df = generate_iot(spark, dgconfig)
df.display()

# COMMAND ----------

df.cache()
temp_null_count = df.where('temperature is null').count()
airflow_null_count = df.where('airflow_rate is null').count()
skyjet_df2 = df.where('model_id="SkyJet234"')
skyjet_df3 = df.where('model_id="SkyJet334"')
defect_rate = df.where('defect=1').count() / df.count()
skyjet2_defect_rate = skyjet_df2.where('defect=1').count() / max(skyjet_df2.count(), 1)
skyjet3_defect_rate = skyjet_df3.where('defect=1').count() / max(skyjet_df3.count(), 1)

# COMMAND ----------

assert(100 * skyjet2_defect_rate < 6)

# COMMAND ----------

assert(100 * skyjet3_defect_rate < 6)

# COMMAND ----------

assert(100 * defect_rate < 3.5)

# COMMAND ----------

assert(temp_null_count == 0)

# COMMAND ----------

assert(airflow_null_count == 0)
