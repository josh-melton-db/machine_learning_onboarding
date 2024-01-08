# Databricks notebook source
# DBTITLE 1,Install DeltaTorch
# MAGIC %pip install pytorch-lightning git+https://github.com/delta-incubator/deltatorch.git

# COMMAND ----------

from utils.onboarding_setup import get_config, generate_iot

config = get_config(spark)
iot_data = generate_iot(spark, 1000, 1) # Use the num_rows or num_devices arguments to change the generated data
train_path = "/dbfs/tmp/jlm/iot_example/train" # If using Unity Catalog, use Volumes instead of dbfs
test_path = "/dbfs/tmp/jlm/iot_example/test/"
BATCH_SIZE = 2048
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/pytorch-distributor'

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id, when, row_number, col


# Sort by timestamp to split chronologically
training_df = iot_data.drop('device_id', 'trip_id', 'timestamp', 'factory_id', 'model_id').orderBy(col('timestamp'))
split_index = int(training_df.count() * 0.7) #use the first 70% of the data for training
training_df = training_df.withColumn("id", row_number().over(Window.orderBy(monotonically_increasing_id()))) # Add incremental id column
train = training_df.where(col("id") <= split_index) # Training on first 70%
test = training_df.where(col("id") > split_index) # Testing on the remaining 30%

train.write.mode("overwrite").option('mergeSchema', 'true').save(train_path.replace('/dbfs', 'dbfs:'))
test.write.mode("overwrite").option('mergeSchema', 'true').save(test_path.replace('/dbfs', 'dbfs:'))
columns = training_df.columns

# COMMAND ----------

import pytorch_lightning as pl
from deltatorch import create_pytorch_dataloader, FieldSpec

class DeltaDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path):
        self.train_path = train_path 
        self.test_path = test_path 
        super().__init__()

    def dataloader(self, path: str, batch_size=BATCH_SIZE):
        return create_pytorch_dataloader(
            path,
            id_field='id',
            fields = [FieldSpec('airflow_rate'), FieldSpec('rotation_speed'), FieldSpec('air_pressure'), 
                      FieldSpec('temperature'), FieldSpec('delay'), FieldSpec('density'), FieldSpec('defect')],
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(self.train_path)

    def test_dataloader(self):
        return self.dataloader(self.test_path)

    def val_dataloader(self):
        return self.dataloader(self.test_path)

# COMMAND ----------

import torch
from torch.utils.data import DataLoader, TensorDataset

features = torch.randn(10, 5)
targets = torch.randn(10, 1)
dataset = TensorDataset(features, targets)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# COMMAND ----------

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import os 
from pyspark.ml.torch.distributor import TorchDistributor

class BinaryClassifier(pl.LightningModule):
    def __init__(self, target_column, input_size, hidden_size=64):
        super().__init__()
        self.target_column = target_column
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        target = batch[self.target_column].float()
        feature_keys = [key for key in batch.keys() if key != self.target_column]
        features = torch.stack([batch[key] for key in feature_keys], dim=1).float()
        predictions = self(features)
        loss = F.binary_cross_entropy(predictions, target.view(-1, 1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch[self.target_column].float()
        feature_keys = [key for key in batch.keys() if key != self.target_column]
        features = torch.stack([batch[key] for key in feature_keys], dim=1).float()
        predictions = self(features)
        loss = F.binary_cross_entropy(predictions, target.view(-1, 1))
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
mlflow.set_experiment(experiment_path)


def train_model(data_module=None, input_size=6, num_gpus=1, single_node=True):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model = BinaryClassifier('defect', input_size)
        model.to(device)
        early_stopping = EarlyStopping(monitor='train_loss', patience=3, mode='min')
        trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping]) #, strategy=strategy, logger=mlflow_logger)
        trainer.fit(model, data_module)
    # os.environ['DATABRICKS_HOST'] = db_host
    # os.environ['DATABRICKS_TOKEN'] = db_token

    # if single_node or num_gpus == 1:
    #     num_devices = num_gpus
    #     num_nodes = 1
    #     strategy="auto"
    # else:
    #     num_devices = 1
    #     num_nodes = num_gpus
    #     strategy = 'ddp_notebook' # check this is ddp or ddp_notebook
    # mlflow.set_experiment(experiment_path)
    # mlflow.pytorch.autolog(disable=False, log_models=False)
    # with mlflow.start_run(run_name='PytorchClassifier') as run:
    #     mlflow_logger = MLFlowLogger(experiment_path) # TODO create a config-based experiment path
    #     early_stopping = EarlyStopping(monitor='train_loss', patience=3, mode='min')
    #     device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #     model = BinaryClassifier('defect', input_size)
    #     model.to(device)
    #     trainer = pl.Trainer(max_epochs=100, callbacks=[early_stopping], strategy=strategy, logger=mlflow_logger)
    #     trainer.fit(model, data_module)
    #     # val_metrics = trainer.validate(model, dataloaders=data_module.val_dataloader(), verbose=False)
    #     # mlflow.log_metrics(val_metrics[0])
    #     if trainer.global_rank == 0:
    #         print('logging model')
    #         reqs = mlflow.pytorch.get_default_pip_requirements() + ["pytorch-lightning==" + pl.__version__]
    #         mlflow.pytorch.log_model(artifact_path="model", pytorch_model=model, pip_requirements=reqs)

input_size = len(training_df.columns) - 2
mlflow.pytorch.autolog(disable=True)
data_module = DeltaDataModule(train_path, test_path)
distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=False)
distributor.run(train_model)

# COMMAND ----------

# DBTITLE 1,Single Node Training
data_module = DeltaDataModule(train_path, test_path)
run = train_model(data_module) # for single node model training

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor

# mlflow.pytorch.autolog(disable=True)
# distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=False)
# distributor.run(train, 1e-3, True)

# COMMAND ----------

pytorch_model = mlflow.pyfunc.load_model(model_uri=f'runs:/{run.info.run_id}/model')
non_defect_predictions = pytorch_model.predict(training_df.where('defect=0').drop('defect', 'id').limit(5).toPandas())
defect_predictions = pytorch_model.predict(training_df.where('defect=1').drop('defect', 'id').limit(5).toPandas())
display(non_defect_predictions)
display(defect_predictions)

# COMMAND ----------


