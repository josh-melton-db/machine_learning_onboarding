import random

def get_config(spark):
  num_rows = random.randint(1000, 2000)
  current_user = spark.sql('select current_user()').collect()[0][0].split('@')[0].replace('.', '_')
  username = spark.sql("SELECT current_user()").first()['current_user()'] 
  schema = f'onboarding'
  return {
      'current_user': current_user,
      'schema' : schema,
      'bronze_table' : f'{schema}.{current_user}_sensor_bronze',
      'defect_table' : f'{schema}.{current_user}_defect_bronze',
      'silver_table' : f'{schema}.{current_user}_sensor_silver',
      'silver_features' : f'{schema}.{current_user}features_table',
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
      'rows_per_run' : num_rows,
      'model_name' : f'device_defect_{current_user}'
  }

def reset_tables(spark, config, dbutils):
  spark.sql(f"drop schema if exists {config['schema']} CASCADE")
  spark.sql(f"create schema {config['schema']}")
  dbutils.fs.rm(config['checkpoint_location'], True)