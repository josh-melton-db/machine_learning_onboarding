# Databricks notebook source
# MAGIC %pip install -q dbldatagen
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from utils.onboarding_setup import iot_generator
iot_df = iot_generator(spark)
iot_df.write.mode('overwrite').saveAsTable('josh_melton.josh_melton_onboarding.test_data')

# COMMAND ----------

final_df = (
    spark.read.table('josh_melton.josh_melton_onboarding.test_data')
)
print(final_df.where('model_id == "SkyJet234"').where('defect == 1').count() / final_df.where('model_id == "SkyJet234"').count())
print(final_df.where('model_id == "SkyJet334"').where('defect == 1').count() / final_df.where('model_id == "SkyJet334"').count())
print(final_df.where('defect == 1').count() / final_df.count())

# COMMAND ----------


