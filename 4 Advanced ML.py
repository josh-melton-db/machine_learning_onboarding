# Databricks notebook source
# MAGIC %md
# MAGIC ### 4. Deep learning, distributed ML
# MAGIC 4.1 Custom MLflow models
# MAGIC
# MAGIC 4.2 Nested runs
# MAGIC
# MAGIC 4.3 Hyperparameter tuning

# COMMAND ----------

import mlflow
import pandas as pd

example_df = pd.DataFrame({'input': [15]})

class Add5(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return 5 + model_input['input']

add5_model = Add5()

with mlflow.start_run() as run:
  mlflow.pyfunc.log_model('model', python_model=add5_model, input_example=example_df)

custom_model = mlflow.pyfunc.load_model(f'runs:/{run.info.run_id}/model')
custom_model.predict(example_df)

# COMMAND ----------

with mlflow.start_run(run_name="Nested Example") as run:
    # Create nested run with nested=True argument
    with mlflow.start_run(run_name="Child 1", nested=True):
        mlflow.log_param("run_name", "child_1")

    with mlflow.start_run(run_name="Child 2", nested=True):
        mlflow.log_param("run_name", "child_2")

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

from hyperopt import SparkTrials

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
