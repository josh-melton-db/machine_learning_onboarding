# Databricks notebook source
pip install -q dbldatagen git+https://github.com/TimeSynth/TimeSynth.git

# COMMAND ----------

import dbldatagen as dg
import dbldatagen.distributions as dist
import pyspark.sql.types
from pyspark.sql.functions import row_number, col, monotonically_increasing_id
import numpy as np
import timesynth as ts
import pandas as pd
from pyspark.sql import Window
import math
from random import random
np.random.seed()

# COMMAND ----------

num_rows = 1000
num_devices = 12

# COMMAND ----------

model_ls = ['TwinX500', 'SkyJet234', 'AeroGlider660', 'ThunderProp890', 'JetLift7000',
            'AirPower360', 'BoostGlide1900', 'FlyForceX550', 'TurboFan3200', 
            'SkySurfer10', 'StormProp760', 'SpeedWing3800', 'FlexJet600', 'JadeEagle890',
            'MightyWing1100', 'EcoJet3000', 'Streamline4150', 'SkyBolt250']
factory_ls = ['A06', 'A18', 'A15', 'A04', 'A10', 'A08', 'A14', 'A03']

# COMMAND ----------

def add_ts_features(pdf: pd.DataFrame) -> pd.DataFrame:
    num_points = len(pdf)
    noise=ts.noise.GaussianNoise(std=0.3)
    less_noise=ts.noise.GaussianNoise(std=0.1)

    sinus_signal=ts.signals.Sinusoidal(amplitude=1.2, frequency=0.35)
    periodic_signal = ts.signals.PseudoPeriodic(amplitude=20, frequency=.35)
    auto_signal = ts.signals.AutoRegressive(ar_param=[1.1, -.5])

    time_sample = ts.TimeSampler(stop_time=20)
    regular_time_samples = time_sample.sample_regular_time(num_points=num_points)

    sinus_ts = ts.TimeSeries(signal_generator=sinus_signal, noise_generator=noise)
    periodic_ts = ts.TimeSeries(signal_generator=periodic_signal, noise_generator=less_noise)
    auto_ts = ts.TimeSeries(signal_generator=auto_signal, noise_generator=less_noise)

    sinusoidal, _, _ = sinus_ts.sample(regular_time_samples)
    pseudo_periodic, _, _ = periodic_ts.sample(regular_time_samples)
    auto_regressive, _, _ = auto_ts.sample(regular_time_samples)
    trend = regular_time_samples*.8
    more_trend = regular_time_samples*1.7

    pdf['rotation_speed'] = auto_regressive
    pdf['temperature'] = (pseudo_periodic + 83) - more_trend
    pdf['delay'] = sinusoidal+trend
    pdf['density'] = sinusoidal + [np.random.randn()*np.sqrt(i) for i, v in enumerate(regular_time_samples)]
    return pdf

schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
            rotation_speed double, pressure double, temperature double, delay float, density float'''

# COMMAND ----------

timestamp_gen = (
    dg.DataGenerator(spark, name="timestamp_data", rows=num_rows)
    .withColumn("timestamp", "timestamp", begin="2023-07-01 00:00:00", end="2023-12-31 23:59:00", random=True)
    .build()
    .withColumn("unique_id", monotonically_increasing_id())
)

device_gen = (
    dg.DataGenerator(sparkSession=spark, name='synthetic_data', rows=num_rows, random=True, randomSeed=num_rows)
    .withColumn('device_id', 'bigint', minValue=1, maxValue=num_devices, random=True)
    .withColumn('model_id', 'string', values=model_ls, base_column='device_id')
    .withColumn('factory_id', 'string', values=factory_ls, base_column='device_id')
    .withColumn('pressure', 'double', minValue=323.0, maxValue=433.53, step=0.01, distribution=dist.Gamma(1.0, 2.0))
    .withColumn('airflow_rate', 'double', minValue=8.0, maxValue=13.3, step=0.1, distribution=dist.Beta(5,1))
    .build()
    .withColumn('device_id', col('device_id').cast('string'))
    .withColumn("unique_id", monotonically_increasing_id())
    .join(timestamp_gen, 'unique_id', 'left').drop('unique_id')
    .withColumn('trip_id', row_number().over(Window.partitionBy('device_id').orderBy('timestamp')))
    .groupBy('device_id')
    .applyInPandas(add_ts_features, schema)
)
device_gen.display()

# COMMAND ----------

def calculate_fourier(seasonal_cycle: np.ndarray, max_cycle: int, n_fourier_terms: int):
    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])

# COMMAND ----------

def add_fourier(pdf, column):
    max_value = pdf[column].max()
    fourier_terms = calculate_fourier(
        pdf[column].astype(float).values,
        max_cycle=max_value,
        n_fourier_terms=1,
    )
    pdf[f'fourier_{column}_sin'] = (pdf[column] / fourier_terms[:, 0]) / 20
    pdf[f'fourier_{column}_cos'] = (pdf[column] / fourier_terms[:, 1]) / 20

# COMMAND ----------

def add_defects(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values(['timestamp'])

    pdf['temp_ewma'] = pdf['temperature'].shift(1).ewm(5).mean()
    pdf['temp_difference'] = pdf['temp_ewma'] - pdf['temperature']
    add_fourier(pdf, 'delay')
    add_fourier(pdf, 'rotation_speed')

    conditions = [
      (pdf['temp_difference'].abs() > 31),
      (pdf['fourier_delay_sin'].abs() > 7.4) | (pdf['fourier_delay_cos'].abs() > 7.4),
      (pdf['fourier_rotation_speed_cos'].abs() > 7.4) | (pdf['fourier_rotation_speed_sin'].abs()>7.4),
      ((pdf['density'].abs() * 6) + pdf['temperature'] > 155),
    ]
    outcomes = [round(random()+.2), round(random()+.3), round(random()+.3), round(random()+.4)]
    pdf['defect'] = np.select(conditions, outcomes)
    pdf = pdf.drop(['fourier_delay_sin', 'fourier_delay_cos', 'fourier_rotation_speed_sin', 
                    'fourier_rotation_speed_cos', 'temp_difference', 'temp_ewma'], axis=1)    
    
    return pdf

defect_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
            rotation_speed double, pressure double, temperature double, delay float, density float, defect float'''
            
final_df = device_gen.groupBy('device_id').applyInPandas(add_defects, defect_schema)
final_df.display()

# COMMAND ----------


