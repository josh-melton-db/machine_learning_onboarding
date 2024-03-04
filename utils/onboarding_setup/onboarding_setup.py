import random
import datetime

def get_config(spark):
  current_user = spark.sql('select current_user()').collect()[0][0].split('@')[0].replace('.', '_')
  username = spark.sql("SELECT current_user()").first()['current_user()'] 
  schema = f'onboarding'
  return {
      'current_user': current_user,
      'schema' : schema,
      'bronze_table' : f'{schema}.{current_user}_sensor_bronze',
      'defect_table' : f'{schema}.{current_user}_defect_bronze',
      'silver_table' : f'{schema}.{current_user}_sensor_silver',
      'silver_features' : f'{schema}.{current_user}_features_table',
      'gold_table' : f'{schema}.{current_user}_sensor_gold',
      'predictions_table' : f'{current_user}_predictions',
      'tuned_bronze_table' : f'{schema}.{current_user}_sensor_bronze_clustered',
      'csv_staging' : f'/{current_user}_onboarding/csv_staging',
      'checkpoint_location' : f'/{current_user}_onboarding/sensor_checkpoints',
      'train_table' : f'/dbfs/tmp/{current_user}/train_table', 
      'test_table' : f'/dbfs/tmp/{current_user}/test_table',  
      'log_path' : f'/dbfs/tmp/{current_user}/pl_training_logger',
      'ckpt_path' : f'/dbfs/tmp/{current_user}/pl_training_checkpoint',
      'experiment_path' : f'/Users/{username}/distributed_pl',
      'model_name' : f'device_defect_{current_user}'
  }

def reset_tables(spark, config, dbutils):
  spark.sql(f"drop schema if exists {config['schema']} CASCADE")
  spark.sql(f"create schema {config['schema']}")
  dbutils.fs.rm(config['checkpoint_location'], True)

dgconfig = {
    "shared": {
        "num_rows": 250000,
        "num_devices": 200,
        "start": datetime.datetime(2023, 1, 1, 0, 0, 0),
        "end": datetime.datetime(2023, 12, 31, 23, 59, 59),
        "frequency": 0.35,
        "amplitude": 1.2,
    },
    "timestamps": {
        "column_name": "timestamp",
        "minimum": 10,
        "maximum": 350,
    },
    "temperature": {
      "lifetime": {
        "column_name": "temperature",
        "noisy": 0.3,
        "trend": 0.1,
        "mean": 58,
        "std_dev": 17,
      },
      "trip": {
        "trend": -0.8,
        "noisy": 1,
      }
    },
    "air_pressure": {
      "depedent_on": "temperature",
      "min": 913,
      "max": 1113,
      "subtract": 15 
    },
    "lifetime": {
        "trend": 0.4,
        "noisy": 0.6,
    },
    "trip": {
        "trend": 0.2,
        "noisy": 1.2,
    },
}