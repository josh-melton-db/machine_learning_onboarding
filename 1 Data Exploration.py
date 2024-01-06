# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction
# MAGIC
# MAGIC Welcome to Onboarding for Machine Learning on Databricks! </br>
# MAGIC Please be sure to use an <a href="https://docs.databricks.com/en/machine-learning/index.html#create-a-cluster-using-databricks-runtime-ml">ML Runtime Cluster</a> with a recent Databricks Runtime. The dataset should be small enough to run on any Databricks cluster, but if you're having trouble use the `num_rows` argument in the `generate_iot()` function to reduce the data volume.
# MAGIC
# MAGIC First we'll install the correct libraries, run the setup, and read our data. You can run cells via the UI or the "shift+enter" hotkey

# COMMAND ----------

# DBTITLE 1,Run Setup
from utils.onboarding_setup import get_config, reset_tables, generate_iot

config = get_config(spark)
reset_tables(spark, config, dbutils)
iot_data = generate_iot(spark) # Use the num_rows or num_devices arguments to change the generated data
iot_data.write.mode('overwrite').saveAsTable(config['bronze_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC #Pandas on Databricks
# MAGIC You can convert the spark dataframe to a pandas dataframe quite easily by using the toPandas() function. Pandas is a single node processing library so we'll start with analysis of one device to reduce the data volume before discussing how to apply pandas in parallel

# COMMAND ----------

# DBTITLE 1,Read Subset and Convert to Pandas
import pandas as pd
bronze_table = spark.read.table(config['bronze_table'])
highest_count_device_id = (
    bronze_table.where('defect=1')
    .groupBy('device_id').count() 
    .orderBy('count', ascending=False)  # Let's tackle the most problematic device in Pandas first, and
).first()[0]                            # later use Spark's distributed processing on the larger dataset
pandas_bronze = bronze_table.where(f'device_id = {highest_count_device_id}').toPandas()

# COMMAND ----------

# DBTITLE 1,Data Exploration in Pandas
print(pandas_bronze.describe())
print(pandas_bronze.columns)
print(pandas_bronze.shape)
print(pandas_bronze.dtypes)
print(pandas_bronze.isnull().sum())

# COMMAND ----------

# DBTITLE 1,Display and Visualize
pandas_bronze = pandas_bronze.sort_values('timestamp')
pandas_bronze['row_num'] = range(1, len(pandas_bronze) + 1)
display(pandas_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC #Notebook Visualizations
# MAGIC - Once you've run the display command in the cell above, try clicking the "+" button to the left of "Table" and explore chart creation. Try creating a Data Profile. 
# MAGIC - Next, create a new visualization and choose a Scatter plot with timestamp or row_num as the X axis and delay, temperature, or density as the Y axis. Group by trip_id or by defect. Use the UI to identify interesting patterns
# MAGIC - Finally, add "defect" to the X axis and count* to the Y axis to see how many defects we're working with
# MAGIC - If a visualization is worth sharing, try the down arrow next to the title to "add to dashboard". Notebook dashboards can be used to collect and share the various visualiztions you create in your notebooks

# COMMAND ----------

# MAGIC %md
# MAGIC #Featurization
# MAGIC First, let's change the categorical factory_id and model_id columns into one hot encodings to make them work for our model training

# COMMAND ----------

# DBTITLE 1,OHE Categoricals
encoded_factory = pd.get_dummies(pandas_bronze['factory_id'], prefix='ohe')
encoded_model = pd.get_dummies(pandas_bronze['model_id'], prefix='ohe')
features = pd.concat([pandas_bronze.drop('factory_id', axis=1).drop('model_id', axis=1), encoded_factory, encoded_model], axis=1)
features = features.drop('timestamp', axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC From our notebook visualizations, we can see that the density data has some seasonality and trend to it. Next let's see if we can turn that into a more straightforward signal for prediction. 
# MAGIC
# MAGIC Create a visualization in the resulting table that shows row_num in the X axis, and the rolling mean column against the original density column in the Y axis. Adjust the "span" parameter to smooth out the wave as best you can. Try dividing the total number of rows by the number of peaks you see in the original data, and be sure to select "render all" in your visualization to see the whole dataset

# COMMAND ----------

# DBTITLE 1,Density EWMA
features['rolling_mean_density'] = features['density'].shift(1).ewm(span=600).mean()
features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's do the same with temperature! By taking the rolling mean, or exponential weighted moving average, of these columns we might add some more predictive power to our model. Maybe sharp deviations from recent temperatures impact the defect rate. Feel free to visualize like we did above. Note that for a single device like we're currently looking at there will likely be breaks in the temperature readings between each trip
# MAGIC
# MAGIC Let's be sure to shift(1) to avoid data leakage

# COMMAND ----------

# DBTITLE 1,EWMA Temperature
features['rolling_mean_temp'] = features['temperature'].shift(1).ewm(5).mean()
features['temp_difference'] = features['rolling_mean_temp'] - features['temperature']
features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we'll drop the row_num column and forward fill null values or replace with zeros where necessary before writing the results to our feature table

# COMMAND ----------

# DBTITLE 1,Fill NAs
features = features.drop('row_num', axis=1)
features = features.fillna(method='ffill')
features = features.fillna(0)

# COMMAND ----------

# DBTITLE 1,Save Features to Table
spark.createDataFrame(features).write.mode('overwrite').saveAsTable(config['silver_features'])

# COMMAND ----------

# MAGIC %md
# MAGIC Looks good! We've now explored our data, shared insights with others, and used the Pandas library that many data scientists are familiar with to create features on our data. Next let's see how Databricks can help us with model experimentation and management
