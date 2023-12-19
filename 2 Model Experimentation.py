# Databricks notebook source
# MAGIC %md
# MAGIC # Introduction

# COMMAND ----------

# DBTITLE 1,Install Libraries
pip install -q dbldatagen

# COMMAND ----------

# DBTITLE 1,Run Setup
from utils.onboarding_setup import get_config, reset_tables, iot_generator
config = get_config(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we've explored our data and added some features in the previous notebook, let's train some models to predict defects! First, we'll split the data into train/test

# COMMAND ----------

# DBTITLE 1,Create Datasets
features = spark.read.table(config['silver_features']).toPandas()

train = features.iloc[:int(len(features) * 0.8)]
test = features.iloc[int(len(features) * 0.8):]

X_train = train.drop('defect', axis=1)
X_test = test.drop('defect', axis=1)
y_train = train['defect']
y_test = test['defect']
X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Experiments
# MAGIC Next we'll try a few different approaches to predicting defects. In order to track the results we'll use MLflow _Experiments_. An Experiment allows you to track and compare many attempts to solve a problem. Each attempt is called a _run_

# COMMAND ----------

# DBTITLE 1,Run MLflow Experiment
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from mlflow.models.signature import infer_signature
import uuid
import matplotlib.pyplot as plt

model_name = f"rf_{config['model_name']}"

with mlflow.start_run(run_name='First Run RF') as run:
    # Create model, train it, and create predictions
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model with signature
    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, model_name, signature=signature)

    # Log metrics
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mlflow.log_metric('test_f1', f1)
    mlflow.log_metric('test_recall', recall)
    mlflow.log_metric('defects_predicted', predictions.sum())

    # Log feature importances plot
    importance = (pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), 
                               columns=["Feature", "Importance"])
                  .sort_values("Importance", ascending=False))
    fig, ax = plt.subplots()
    importance.plot.bar(x='Feature', ax=ax)
    plt.title("Feature Importances")
    mlflow.log_figure(fig, "feature_importances.png")

# COMMAND ----------

# MAGIC %md
# MAGIC You can view the experiment by selecting the beaker icon on the right side of the screen, or choosing "Experiments" from the left menu and selecting your experiment that matches the name of this notebook. Try clicking around to see where the model information resides, how to make charts of the information logged, and where the artifacts such as the feature importance chart can be accessed. Look at the metrics that were logged and see if the low proportion of defects may have thrown off our model. Since the opportunity cost of missing a defect is very high, recall will be an important metric for us. Let's upsample our training data before the next run. Be careful to avoid data leakage when doing this! We'll try Synthetic Minority Oversampling (SMOTE)

# COMMAND ----------

# DBTITLE 1,Upsample Data
from imblearn.over_sampling import SMOTE
from collections import Counter

counter1 = Counter(y_train)
oversample = SMOTE()
X_train_oversampled, y_train_oversampled = oversample.fit_resample(X_train, y_train)
counter2 = Counter(y_train_oversampled)
print(counter1, counter2)

# COMMAND ----------

# MAGIC %md
# MAGIC As a part of our experiment, let's try another run with the exact same code but swap in our upsampled training data

# COMMAND ----------

# DBTITLE 1,Second Run
with mlflow.start_run(run_name='Second Run RF') as run:
    # Create model, train it, and create predictions
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_oversampled, y_train_oversampled)
    predictions = rf.predict(X_test)

    # Log model with signature
    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, model_name, signature=signature)

    # Log metrics
    f1 = f1_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mlflow.log_metric('test_f1', f1)
    mlflow.log_metric('test_recall', recall)
    mlflow.log_metric('defects_predicted', predictions.sum())

    # Log feature importances plot
    importance = (pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), 
                               columns=["Feature", "Importance"])
                  .sort_values("Importance", ascending=False))
    fig, ax = plt.subplots()
    importance.plot.bar(x='Feature', ax=ax)
    plt.title("Feature Importances")
    mlflow.log_figure(fig, "feature_importances.png")

# COMMAND ----------

# MAGIC %md
# MAGIC Our F1 score dropped, but recall improved! We may need to balance the two based on a cost-benefit analysis moving forward, but luckily we're tracking all of our runs and can select the model that turns out to be the best fit later!
# MAGIC
# MAGIC Let's try one more time, this time using MLflow's autolog() capability to log the model and numerous defaults without adding extra code

# COMMAND ----------

# DBTITLE 1,Third Run
from sklearn.linear_model import LogisticRegression

model_name = f"lr_{config['model_name']}"
mlflow.sklearn.autolog() # Autolog creates the run and adds the important information for us

# Create model, train it, and create predictions. Defer logging to autolog()
lr = LogisticRegression()
lr.fit(X_train_oversampled, y_train_oversampled)
predictions = lr.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Registry
# MAGIC Now that we've had a chance to compare three models, let's determine the best one and add it to the model registry for downstream use

# COMMAND ----------

# DBTITLE 1,Find Best Model
from mlflow.tracking import MlflowClient
client = MlflowClient()

runs = client.search_runs(run.info.experiment_id, order_by=["metrics.recall DESC"])
lowest_f1_run_id = runs[0].info.run_id

# COMMAND ----------

# DBTITLE 1,Register Model
model_uri = f"runs:/{lowest_f1_run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=config['model_name'])

# COMMAND ----------

# MAGIC %md
# MAGIC Once we feel confident in our model's predictions, we can change the stage to Staging or Production so downstream consumers can use it. As we'll see in the next notebook, if we want to change to different model libraries our downstream consumers will be able to seamlessly continue to use the model

# COMMAND ----------

# DBTITLE 1,Transition to Production
client.transition_model_version_stage(
    name=config['model_name'],
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------


