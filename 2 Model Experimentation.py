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
# MAGIC Now that we've explored our data and added some features in the previous notebook, let's train some models to predict defects! First, we'll import the standard SKlearn library to do our test train split

# COMMAND ----------

from sklearn.model_selection import train_test_split

features = spark.read.table(config['silver_features']).toPandas()
X_train, X_test, y_train, y_test = train_test_split(features.drop(["defect"], axis=1), features["defect"], random_state=1)
X_train.head()

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from mlflow.models.signature import infer_signature
import uuid
import matplotlib.pyplot as plt

model_name = "random_forest_" + str(uuid.uuid1())[:5] # generate a unique name for your mlflow model

with mlflow.start_run(run_name='First RF Run') as run:
    # Create model, train it, and create predictions
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model with signature
    signature = infer_signature(X_train, predictions)
    mlflow.sklearn.log_model(rf, model_name, signature=signature)

    # Log metrics
    f1 = f1_score(y_test, predictions)
    mlflow.log_metric('f1', f1)

    # Log feature importances plot
    importance = (pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), columns=["Feature", "Importance"])
                  .sort_values("Importance", ascending=False))
    fig, ax = plt.subplots()
    importance.plot.bar(x='Feature', ax=ax)
    plt.title("Feature Importances")
    plt.savefig("feature_importances.png")  # Save the figure to a file
    plt.close(fig)
    mlflow.log_figure(fig, "feature_importances.png")

    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    print(f"Inside MLflow Run with run_id `{run_id}` and experiment_id `{experiment_id}`")

# COMMAND ----------


