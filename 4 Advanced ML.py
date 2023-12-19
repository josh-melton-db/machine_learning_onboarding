# Databricks notebook source
# MAGIC %md
# MAGIC # Advanced ML
# MAGIC This repository is incomplete and for outline purposes only

# COMMAND ----------

pip install -q dbldatagen

# COMMAND ----------

from utils.onboarding_setup import get_config, reset_tables, iot_generator
config = get_config(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC #Pandas UDFs
# MAGIC So far we've used pandas to run some single-node transformations on our data. If our data volume grows, we may want to run processes in parallel instead. Databricks offers three integrations with Spark to do this:
# MAGIC - Pyspark Pandas
# MAGIC - Pandas UDFs
# MAGIC - Apply In Pandas
# MAGIC
# MAGIC Let's test the pyspark pandas approach to run pandas transformations in parallel

# COMMAND ----------

import pyspark.pandas as ps

features = spark.read.table(config['silver_features']).pandas_api()
ewma = features['temperature'].shift(1).ewm(5).mean()
ewma

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
# MAGIC     mlflow.pyfunc.log_model('model', python_model=Add5(), input_example=example_df)
# MAGIC ```
# MAGIC </br>
# MAGIC Let's use a custom MLflow model to combine some different approaches to solving our prediction problem

# COMMAND ----------

import mlflow
import pandas as pd

example_df = pd.DataFrame({'input': [15]})

class Add5(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return 5 + model_input['input']
        # Integrate ARIMA

add5_model = Add5()

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model('model', python_model=add5_model, input_example=example_df)

custom_model = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/model')
custom_model.predict(example_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nested MLflow models
# MAGIC Other times we want our experimentation to be nested. As an example, we may want to create a different ML model for each model of engine to account for their different responses to our features. Let's try logging multiple runs within a parent run

# COMMAND ----------

with mlflow.start_run(run_name="Nested Example") as run:
    # Create nested run with nested=True argument
    with mlflow.start_run(run_name="Child 1", nested=True):
        mlflow.log_param("run_name", "child_1")

    with mlflow.start_run(run_name="Child 2", nested=True):
        mlflow.log_param("run_name", "child_2")

# COMMAND ----------

# MAGIC %md
# MAGIC # Hyperparameter tuning
# MAGIC Now that we've done our experimentation and tracked the building of so many models, we may have an idea of the best type of model but want to ensure our hyperparameter selection for that type of model is optimal. We can do that with hyperopt, a framework where we can minimize the output of some function given a parameter space to explore as input

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials

# Define objective function
def objective(params):
    model = RandomForestRegressor(n_estimators=int(params["n_estimators"]), 
                                  max_depth=int(params["max_depth"]), 
                                  min_samples_leaf=int(params["min_samples_leaf"]),
                                  min_samples_split=int(params["min_samples_split"]))
    model.fit(X_train, y_train)
    pred = model.predict(X_train)
    score = mean_squared_error(pred, y_train)

    # Hyperopt minimizes score, here we minimize mse. 
    return score

# COMMAND ----------

# Define search space
search_space = {"n_estimators": hp.quniform("n_estimators", 100, 500, 5),
                "max_depth": hp.quniform("max_depth", 5, 20, 1),
                "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 5, 1),
                "min_samples_split": hp.quniform("min_samples_split", 2, 6, 1)}

# Set parallelism (should be order of magnitude smaller than max_evals)
spark_trials = SparkTrials(parallelism=2)

with mlflow.start_run(run_name="Hyperopt"):
    argmin = fmin(fn=objective,
                  space=search_space,
                  algo=tpe.suggest,
                  max_evals=16,
                  trials=spark_trials)

# COMMAND ----------


