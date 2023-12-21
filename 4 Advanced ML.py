# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced ML
# MAGIC This notebook is incomplete and for outline purposes only

# COMMAND ----------

# DBTITLE 1,Install Libraries
pip install -q dbldatagen

# COMMAND ----------

# DBTITLE 1,Import Config
from utils.onboarding_setup import get_config
config = get_config(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC # Pandas + Spark
# MAGIC So far we've used pandas to run some single-node transformations on our data. If our data volume grows, we may want to run processes in parallel instead. Spark offers several approaches including
# MAGIC - Pyspark Pandas
# MAGIC - Pandas UDFs
# MAGIC - Apply In Pandas
# MAGIC
# MAGIC TODO: add links for each
# MAGIC
# MAGIC Let's try the applyinpandas approach to run pandas our transformation in parallel across all of the trips in our dataset. This will scale out to all the nodes on our spark cluster, as opposed to traditional pandas which is single node and will encounter OOM errors at scale

# COMMAND ----------

# DBTITLE 1,Pandas Features in Parallel
import pandas as pd
features = spark.read.table(config['bronze_table'])

def add_rolling_temp(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf['rolling_mean_temp'] = pdf['temperature'].ewm(5).mean()
    pdf['temp_difference'] = pdf['rolling_mean_temp'] - pdf['temperature']
    pdf = pdf.fillna(method='ffill').fillna(0)
    return pdf

rolling_temp_schema = '''
    device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
    rotation_speed double, pressure double, temperature double, delay float, density float, defect float,
    rolling_mean_temp double, temp_difference double
'''

features = features.orderBy('timestamp')
features = features.groupBy('trip_id').applyInPandas(add_rolling_temp, rolling_temp_schema)
features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's do the same, but per device for the rolling density metric

# COMMAND ----------

# DBTITLE 1,Trip Features in parallel
def add_rolling_density(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf['rolling_mean_density'] = pdf['density'].shift(1).ewm(span=600).mean()
    pdf = pdf.fillna(method='ffill').fillna(0)
    return pdf

rolling_density_schema = '''
    device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
    rotation_speed double, pressure double, temperature double, delay float, density float, defect float,
    rolling_mean_temp double, temp_difference double, rolling_mean_density double
'''

features = features.groupBy('device_id').applyInPandas(add_rolling_density, rolling_density_schema)
features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC If you chart visualize the temperature for defects vs non-defects, you can see that temperature has a significant impact on defect rate. Currently we're using a rolling mean to provide our model more informative temperature features. But what if we could integrate a forward looking temperature prediction into the features?
# MAGIC
# MAGIC Let's train an ARIMA model to predict the next temperature that will occur using another method of parallelizing pandas operations, a Pandas UDF

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, lit
from statsmodels.tsa.arima.model import ARIMA


@pandas_udf("double")
def forecast_arima(temperature: pd.Series, order_series: pd.Series) -> pd.Series:
    order = tuple(map(int, order_series.iloc[0].strip('()').split(',')))
    model = ARIMA(temperature, order=order)
    model_fit = model.fit()
    return model_fit.predict()

# Minimal Spark code - just select one column and add another. Our library, along with our complex logic, is still in pandas
temp_predictions = features.select('temperature').withColumn('predicted_temp', forecast_arima('temperature', lit("(1,2,4)")))
temp_predictions.display()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="predicted_temp", labelCol="temperature", metricName="rmse")
rmse = evaluator.evaluate(temp_predictions)
rmse

# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameter tuning
# MAGIC Now we've seen how to run ARIMA in parallel on a large dataset, but we haven't determined which hyperparameters (the "order" parameter) gives the best ARIMA model for our use case. We can explore the correct hyperparameters by using hyperopt, a framework where we can minimize the output of some function given a parameter space to explore as input. In our case, we'll turn the prediction and rmse calculations into our objective function, and use hyperopt to automatically and intelligently explore many values for the "order" hyperparameters.

# COMMAND ----------

# DBTITLE 1,ARIMA Model Definition
train_arima_features = features.select('temperature').where('model_id="SkyJet334"')

# Define objective function to minimize
def objective(params):
    order = str((int(params["p"]), int(params["d"]), int(params["q"])))
    temp_predictions = train_arima_features.withColumn('predicted_temp', forecast_arima('temperature', lit(order)))
    return evaluator.evaluate(temp_predictions)

# Test two runs of the objective function with different parameters. Lower is better with the rmse evaluator
print(objective({'p': 1, 'd': 1, 'q': 1}), objective({'p': 1, 'd': 2, 'q': 0}))

# COMMAND ----------

# DBTITLE 1,Hyperparameter Tune ARIMA
from hyperopt import fmin, tpe, hp, SparkTrials
import mlflow 

# Define search space. 64 possibilities, but Hyperopt identifies the best combinations to try
search_space = {'p': hp.quniform('p', 0, 4, 1),
                'd': hp.quniform('d', 0, 2, 1),
                'q': hp.quniform('q', 0, 5, 1)}

# Run intelligent hyperparameter search over the search space
# This may take a few minutes - you can reduce max_evals to speed it up
argmin = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=16)
print(argmin)

# COMMAND ----------

# DBTITLE 1,Train Optimal ARIMA
# # Use our optimal ARIMA hyperparameters and add the predictions as a feature
# optimal_order = str((int(argmin['p']), int(argmin['d']), int(argmin['q'])))
# features = features.withColumn('predicted_temp', forecast_arima('temperature', lit(optimal_order)))
# features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's finally train a new version of a classifier model from the experimentation notebook using the ARIMA model's output as a feature. We'll get back to using the full dataset soon

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

# Use our optimal ARIMA hyperparameters and add the predictions as a feature
optimal_order = str((int(argmin['p']), int(argmin['d']), int(argmin['q'])))
arima_features = spark.read.table(config['silver_features']).withColumn('predicted_temp', forecast_arima('temperature', lit(optimal_order)))
arima_features = arima_features.toPandas()

train = arima_features.iloc[:int(len(arima_features) * 0.8)]
test = arima_features.iloc[int(len(arima_features) * 0.8):]

X_train = train.drop('defect', axis=1)
X_test = test.drop('defect', axis=1)
y_train = train['defect']
y_test = test['defect']

mlflow.sklearn.autolog()
with mlflow.start_run(run_name='ARIMA Features') as run:
    rf = RandomForestClassifier(n_estimators=100) # We could run hyperopt for these hyperparameters too!
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    recall = recall_score(y_test, predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC # Custom MLflow Models
# MAGIC Sometimes an out of the box model from one of the libraries MLflow integrates won't get the job done, so you need to do something custom. Adding last-second transformations to inputs or outputs of a model, or combining the results of two different models as a single model might stop you from using a standard MLflow compatible library. Luckily, there's a way to create custom models in MLflow by subclassing PythonModel and including a predict() function like below:
# MAGIC </br></br>
# MAGIC ```
# MAGIC class Add5(mlflow.pyfunc.PythonModel):
# MAGIC     def predict(self, context, model_input):
# MAGIC         return 5 + model_input['feature_col'] 
# MAGIC
# MAGIC with mlflow.start_run() as run:
# MAGIC     mlflow.pyfunc.log_model('model', python_model=Add5())
# MAGIC ```
# MAGIC
# MAGIC What happens when we want to make an update to the ARIMA model? Or when we change our feature logic? If the defect model was trained with a different input, it may produce incorrect results. Let's use a custom MLflow model to make sure that our defect model always gets paired with the the intended ARIMA model and features

# COMMAND ----------

class ComboModel(mlflow.pyfunc.PythonModel):
    def __init__(self, optimal_order, defect_model, temp_func, temp_schema, density_func, density_schema, forecast_arima):
        self.order = optimal_order
        self.defect_model = defect_model
        self.temp_func = temp_func
        self.temp_schema = temp_schema
        self.density_func = density_func
        self.density_schema = density_schema
        self.forecast_arima = forecast_arima

    def generate_features(self, df):
        features = df.withColumn('predicted_temp', self.forecast_arima('temperature', lit(self.order)))
        features = features.groupBy('trip_id').applyInPandas(self.temp_func, self.temp_schema)
        features = features.groupBy('device_id').applyInPandas(self.density_func, self.density_schema)
        return features

    def predict(self, context, model_input):
        return self.defect_model.predict(model_input['input'])

combo_model = ComboModel(optimal_order, rf, add_rolling_temp, rolling_temp_schema, add_rolling_density, rolling_density_schema, forecast_arima)

# COMMAND ----------

raw = spark.read.table(config['bronze_table'])
combo_features = combo_model.generate_features(raw)
combo_features.display()

# COMMAND ----------

# MAGIC %md
# MAGIC By logging the feature logic along with our model, we've eliminated a lot of potential headaches in productionalizing our model, such as online/offline skew. Now, let's log the custom model to MLflow

# COMMAND ----------

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model('combo_model', python_model=combo_model, input_example=example_df)

custom_model = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/combo_model')
custom_model.predict(example_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nested MLflow models
# MAGIC Try reading the entire dataset and displaying the fault rate per model_id. The defect rates are different for each model_id - maybe we should consider training different ML models to make sure our predictions are aligned to the factors at play for each type of device (model_id) rather than making a very generalized ML model for all model_ids
# MAGIC
# MAGIC This is a great scenario for our experimentation to be nested. We want to create a different ML model for each model of engine to account for their different responses to our features. Let's try logging multiple runs within a parent run, and as an extra challenge we'll do it in parallel

# COMMAND ----------

with mlflow.start_run(run_name="Nested Example") as run:
    # Create nested run with nested=True argument
    with mlflow.start_run(run_name="Child 1", nested=True):
        mlflow.log_param("run_name", "child_1")

    with mlflow.start_run(run_name="Child 2", nested=True):
        mlflow.log_param("run_name", "child_2")

# COMMAND ----------


