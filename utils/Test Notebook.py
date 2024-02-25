# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

from onboarding_setup import generate_iot

df = generate_iot(spark)
df.cache()
skyjet_df2 = df.where('model_id="SkyJet234"')
skyjet_df3 = df.where('model_id="SkyJet334"')
print(100 * (df.where('defect=1').count() / df.count()))
print(100 * (skyjet_df2.where('defect=1').count() / skyjet_df2.count()))
print(100 * (skyjet_df3.where('defect=1').count() / skyjet_df3.count()))
print(df.where('temperature is null').count(), df.where('airflow_rate is null').count())

# COMMAND ----------


