# Databricks notebook source
# MAGIC %md
# MAGIC # 5 Distributed Deep Learning With Pytorch Lightning
# MAGIC While our dataset isn't complex enough to warrant deep learning, some problems might require more sophisticated solutions. Distributing deep learning across nodes is complex, but luckily we've got tooling that plugs the common deep learning frameworks (Pytorch and Tensorflow) into Databricks' stack (Spark and Delta Lake) quite simply. 
# MAGIC
# MAGIC In this notebook, we'll create a dataloader out of a Delta Lake table and define a [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/) model, then use those to create a model training function. We'll use that model training function to train the model on a single node, and then use the exact same function to distribute the training of the model. This is helpful when you've got large volumes of data which require

# COMMAND ----------

# DBTITLE 1,Install DeltaTorch and PL
# MAGIC %pip install pytorch-lightning git+https://github.com/delta-incubator/deltatorch.git

# COMMAND ----------

# DBTITLE 1,Setup
from utils.onboarding_setup import get_config, generate_iot
import mlflow

config = get_config(spark)
iot_data = generate_iot(spark, 1000, 1) # Use the num_rows or num_devices arguments to change the generated data
train_path = "/dbfs/tmp/jlm/iot_example/train" # If using Unity Catalog, use Volumes instead of dbfs
test_path = "/dbfs/tmp/jlm/iot_example/test/" # TODO: convert to delta table
BATCH_SIZE = 2048
username = spark.sql("SELECT current_user()").first()['current_user()'] # TODO: turn into config
experiment_name = 'pytorch_distributor'
experiment_path = f'/Users/{username}/{experiment_name}'
log_path = f"/dbfs/Users/{username}/pl_training_logger"
ckpt_path = f"/dbfs/Users/{username}/pl_training_checkpoint"

# COMMAND ----------

# DBTITLE 1,Create Features
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id, when, row_number, col


# Sort by timestamp to split chronologically
training_df = iot_data.drop('device_id', 'trip_id', 'timestamp', 'factory_id', 'model_id').orderBy(col('timestamp'))
# split_index = int(training_df.count() * 0.7) #use the first 70% of the data for training
# training_df = training_df.withColumn("id", row_number().over(Window.orderBy(monotonically_increasing_id()))) # Add incremental id column
# train = training_df.where(col("id") <= split_index) # Training on first 70%
# test = training_df.where(col("id") > split_index) # Testing on the remaining 30%

# train.write.mode("overwrite").option('mergeSchema', 'true').save(train_path.replace('/dbfs', 'dbfs:'))
# test.write.mode("overwrite").option('mergeSchema', 'true').save(test_path.replace('/dbfs', 'dbfs:'))
input_columns = training_df.drop('id').columns

# COMMAND ----------

# DBTITLE 1,Dataloader
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
            fields = [FieldSpec(field) for field in input_columns],
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(self.train_path)

    def test_dataloader(self):
        return self.dataloader(self.test_path)

    def val_dataloader(self):
        return self.dataloader(self.test_path)

# COMMAND ----------

# DBTITLE 1,Model Definition
import torch
from torch.utils.data import DataLoader, TensorDataset
from pyspark.ml.torch.distributor import TorchDistributor
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping
import mlflow
import os


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
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # log to mlflow
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# COMMAND ----------

# DBTITLE 1,Model Training Function
db_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().extraContext().apply('api_url')
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def train_model(dataloader, input_size, num_gpus=1, single_node=True):
    os.environ['DATABRICKS_HOST'] = db_host
    os.environ['DATABRICKS_TOKEN'] = db_token
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = BinaryClassifier('defect', input_size)
    model.to(device)
    mlflow.autolog(disable=True)
    mlflow.set_experiment(experiment_path)
    logger = MLFlowLogger(experiment_name=experiment_path)
    early_stopping = EarlyStopping(monitor='train_loss', patience=3, mode='min', log_rank_zero_only=True)
    trainer = pl.Trainer(max_epochs=5, logger=logger, callbacks=[early_stopping], default_root_dir=log_path)
    trainer.fit(model, dataloader)
    return model

# COMMAND ----------

# DBTITLE 1,Set Up Distributors
input_size = len(input_columns) - 1
distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=False)
data_module = DeltaDataModule(train_path, test_path)

# COMMAND ----------

# DBTITLE 1,Single Node Run
model = train_model(data_module, input_size)

# COMMAND ----------

# DBTITLE 1,Multi Node Run
model = distributor.run(train_model, data_module, input_size)

# COMMAND ----------


