# Databricks notebook source
# MAGIC %md
# MAGIC #Model Experimentation

# COMMAND ----------

# DBTITLE 1,Install Libraries
pip install -q dbldatagen git+https://github.com/TimeSynth/TimeSynth.git

# COMMAND ----------

# DBTITLE 1,Run Setup
from utils.onboarding_setup import get_config, reset_tables, iot_generator
config = get_config(spark)

# COMMAND ----------

# MAGIC %md
# MAGIC #####     2.1 Model / system metrics
# MAGIC #####     2.2 Visualize metrics

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we've explored our data and added some features in the previous notebook, let's train some models to predict defects! First, we'll split the data into train/test

# COMMAND ----------

features = spark.read.table(config['silver_features']).toPandas()
features = features.sort_values(by='trip_id')

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
# MAGIC Next we'll try a few different approaches to predicting defects. In order to track the results we'll use MLflow _Experiments_. Experiments allow you to track and compare many attempts to solve a problem. Each attempt is called a _run_

# COMMAND ----------

import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(rf, model_name, signature=signature)

    # Log metrics
    f1 = f1_score(y_test, predictions)
    mlflow.log_metric('f1', f1) # TODO: make the results less bad

    # Log feature importances plot
    importance = (pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), columns=["Feature", "Importance"])
                  .sort_values("Importance", ascending=False))
    fig, ax = plt.subplots()
    importance.plot.bar(x='Feature', ax=ax)
    plt.title("Feature Importances")
    plt.savefig("feature_importances.png")  # Save the figure to a file
    plt.close(fig)
    mlflow.log_figure(fig, "feature_importances.png")

# COMMAND ----------

# MAGIC %md
# MAGIC You can view the experiment by selecting the beaker icon on the right side of the screen, or choosing "Experiments" from the left menu and selecting your experiment that matches the name of this notebook. Try clicking around to see where the model information resides, how to make charts of the information logged, and where the artifacts such as the feature importance chart can be accessed. Looking at the metrics that were logged it seems like the low proportion of defects may have thrown off our model. Let's upsample our training data before training the next one. Be careful to avoid data leakage when doing this!

# COMMAND ----------

features_majority = train[train['defect']!=1]
features_minority = train[train['defect']==1]
features_upsample = resample(features_minority, replace=True, n_samples=len(features_majority))

train_upsampled = pd.concat([features_majority, features_upsample])
X_train_upsampled = train_upsampled.drop('defect', axis=1)
y_train_upsampled = train_upsampled['defect']

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


model_name = f"tree_{config['model_name']}"

with mlflow.start_run(run_name='Second Run Tree') as run:
    # Create model, train it, and create predictions
    tree = DecisionTreeClassifier()
    tree.fit(X_train_upsampled, y_train_upsampled)
    predictions = tree.predict(X_test)

    # Log model with signature
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(tree, model_name, signature=signature)
    mlflow.log_param('upsampled', 'true')

    # Log metrics
    f1 = f1_score(y_test, predictions)
    mlflow.log_metric('f1', f1)

    # Log feature importances plot
    importance = (pd.DataFrame(list(zip(X_train.columns, tree.feature_importances_)), columns=["Feature", "Importance"])
                  .sort_values("Importance", ascending=False))
    fig, ax = plt.subplots()
    importance.plot.bar(x='Feature', ax=ax)
    plt.title("Feature Importances")
    plt.savefig("feature_importances.png")  # Save the figure to a file
    plt.close(fig)
    mlflow.log_figure(fig, "feature_importances.png")

# COMMAND ----------

# MAGIC %md
# MAGIC To skip adding code to explicitly log your models, try MLflow's autolog() capability

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

model_name = f"lr_{config['model_name']}"
mlflow.sklearn.autolog() # Autolog create the run and adds the important information for us
# lr = LogisticRegression()
# lr.fit(X_train, y_train) # TODO reorganize to make this outside the context manager

with mlflow.start_run(run_name='Third Run LR') as run:
    # Create model, train it, and create predictions. Defer logging to autolog() apart from our f1 metric for comparison
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    f1 = f1_score(y_test, predictions)
    mlflow.log_metric('f1', f1)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we've had a chance to compare three models, let's determine the best one and add it to the model registry for downstream use

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

# COMMAND ----------

model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Once we feel confident, we can change the stage to Staging or Production so downstream consumers can use it

# COMMAND ----------

client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------


