# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC Welcome to an Overview of Machine Learning on Databricks! </br>
# MAGIC First we'll install the correct libraries, run the setup, and read our data. You can run cells via the UI or the "shift+enter" hotkey

# COMMAND ----------

# DBTITLE 1,Install Libraries
pip install -q dbldatagen git+https://github.com/TimeSynth/TimeSynth.git

# COMMAND ----------

# DBTITLE 1,Run Setup
from utils.onboarding_setup import get_config, reset_tables, iot_generator

config = get_config(spark)
reset_tables(spark, config, dbutils)
iot_data = iot_generator(spark, config['rows_per_run'])
iot_data.write.mode('overwrite').saveAsTable(config['bronze_table'])

# COMMAND ----------

# DBTITLE 1,Read Data
bronze_df = spark.read.table(config['bronze_table'])
bronze_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook Visualizations
# MAGIC - Once you've run the read and display commands, try clicking the "+" button to the left of "Table" and explore chart creation. Try creating a Data Profile. 
# MAGIC - Next, create a new visualization and choose a Scatter plot with timestamp as the X axis and delay, temperature, or density as the Y axis. Group by device_id. Use the UI to identify interesting patterns
# MAGIC - Finally, add "defect" to the X axis and count* to the Y axis to see how many defects we're working with
# MAGIC - If a visualization is worth sharing, try the down arrow next to the title to "add to dashboard". Notebook dashboards can be used to collect and share the various visualiztions you create in your notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC #Pandas on Databricks
# MAGIC You can convert the spark dataframe to a pandas dataframe quite easily by using the toPandas() function. Pandas is a single node processing library so we'll start with analysis of one device to reduce the data volume, before discussing how to apply pandas in parallel

# COMMAND ----------

# DBTITLE 1,Convert to Pandas
import pandas as pd
pandas_bronze = bronze_df.where('device_id=1').toPandas()

# COMMAND ----------

# DBTITLE 1,Data Exploration in Pandas
print(pandas_bronze.describe())
print(pandas_bronze.columns)
print(pandas_bronze.shape)
print(pandas_bronze.dtypes)
print(pandas_bronze.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC #Featurization
# MAGIC First, let's change the categorical factory_id and model_id columns into one hot encodings to make then work for our model training

# COMMAND ----------

encoded_factory = pd.get_dummies(pandas_bronze['factory_id'], prefix='ohe')
encoded_model = pd.get_dummies(pandas_bronze['model_id'], prefix='ohe')
features = pd.concat([pandas_bronze.drop('factory_id', axis=1).drop('model_id', axis=1), encoded_factory, encoded_model], axis=1)
features = features.drop('timestamp', axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC From our initial exploration and visualization, we can see that the delay data has some seasonality and trend to it. Next let's see if we can turn that into a more straightforward signal for prediction. 
# MAGIC
# MAGIC Create a visualization in the resulting table that shows the rolling mean column against the original delay column and adjust the "span" parameter to smooth out the wave as best you can.

# COMMAND ----------

features = features.sort_values(by='trip_id')
features['rolling_mean_delay'] = features['delay'].shift(1).ewm(span=12).mean()
features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's do the same with temperature! By taking the rolling mean, or exponential weighted moving average, of these columns we might add some more predictive power to our model. Maybe sharp deviations from recent temperatures impact the defect rate. 
# MAGIC
# MAGIC Let's be sure to shift(1) to avoid data leakage

# COMMAND ----------

features['rolling_mean_temp'] = features['temperature'].shift(1).ewm(5).mean()
features['temp_difference'] = features['rolling_mean_temp'] - features['temperature']
features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we'll forward fill null values or replace with zeros where necessary before writing the results to our feature table

# COMMAND ----------

features = features.fillna(method='ffill')
features = features.fillna(0)

# COMMAND ----------

# DBTITLE 1,Save Features to Table
spark.createDataFrame(features).write.mode('overwrite').saveAsTable(config['silver_features'])

# COMMAND ----------

# MAGIC %md
# MAGIC Looks good! We've now explored our data, shared insights with others, and used the Pandas library that many data scientists are familiar with to create features on our data. Next let's see how Databricks can help us with model experimentation and management

# COMMAND ----------


