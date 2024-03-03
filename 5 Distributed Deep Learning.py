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
from utils.onboarding_setup import get_config

config = get_config(spark)
BATCH_SIZE = 2048
EPOCHS = 20

# COMMAND ----------

# MAGIC %md
# MAGIC DeltaTorch requires a unique id column, so we'll add that before we save to our target feature tables. For purposes of demonstration we'll also stick to simple numeric columns
# MAGIC

# COMMAND ----------

# DBTITLE 1,Create Features
from pyspark.sql import Window
from pyspark.sql.functions import monotonically_increasing_id, row_number, col, lit

bronze_df = spark.read.table(config['bronze_table'])
categorical_cols = ['device_id', 'trip_id', 'timestamp', 'factory_id', 'model_id']
training_df = bronze_df.drop(*categorical_cols).orderBy(col('timestamp'))
training_df = training_df.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())))
training_cols = training_df.drop('id').columns

split_index = int(training_df.count() * 0.7) 
train_df = training_df.where(col('id') <= split_index)
test_df = training_df.where(col('id') > split_index)
train_df.write.mode('overwrite').format('delta').save(config['train_table'].replace('/dbfs', 'dbfs:')) 
test_df.write.mode('overwrite').format('delta').save(config['test_table'].replace('/dbfs', 'dbfs:'))

# COMMAND ----------

# MAGIC %md
# MAGIC Once we've got our features into Delta and noted the path to the table, we're ready to create our dataloader class

# COMMAND ----------

# DBTITLE 1,Dataloader Definition
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
            fields = [FieldSpec(field) for field in training_cols],
            batch_size=batch_size,
        )

    def train_dataloader(self):
        return self.dataloader(self.train_path)

    def test_dataloader(self):
        return self.dataloader(self.test_path)

    def val_dataloader(self):
        return self.dataloader(self.test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll create our PyTorch Lightning model class and set up our experiment. Note that this paradigm is the same for vanilla PyTorch, and very similar for Tensorflow

# COMMAND ----------

# DBTITLE 1,Model Definition
import mlflow
import os
from torch.utils.data import DataLoader, TensorDataset
from pyspark.ml.torch.distributor import TorchDistributor
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping


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
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True) # logs to MLflow
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
    mlflow.set_experiment(config['experiment_path'])
    logger = MLFlowLogger(experiment_name=config['experiment_path'])
    early_stopping = EarlyStopping(monitor='train_loss', patience=3, mode='min', log_rank_zero_only=True)
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=logger, callbacks=[early_stopping], default_root_dir=config['log_path'])
    trainer.fit(model, dataloader)
    return model

# COMMAND ----------

# DBTITLE 1,Create Dataloader
input_size = len(training_cols) - 1 # all columns minus the label
data_module = DeltaDataModule(config['train_table'], config['test_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's try running our train_model function on a single node by simply passing it the delta module and the input size

# COMMAND ----------

# DBTITLE 1,Single Node Run
model = train_model(data_module, input_size)

# COMMAND ----------

# MAGIC %md
# MAGIC If we've got more data than a single node can handle, we can try distributing the training run across our node. `num_processes` is the parameter that controls the level of parallelism - we can set this to the number of gpus or nodes our cluster has. We highly recommend starting with single node training runs and only introducing the complexity of distributed deep learning if those don't work. If you're using the default settings and a typical cluster, the single node training run will most likely be faster in this instance since it doesn't incur the overhead of communication between nodes. However, if you're working against constraints such as a massive dataset that won't fit into memory, this can be a more efficient approach

# COMMAND ----------

# DBTITLE 1,Multi Node Run
distributor = TorchDistributor(num_processes=2, local_mode=False, use_gpu=False)
model = distributor.run(train_model, data_module, input_size)

# COMMAND ----------


