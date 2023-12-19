import dbldatagen as dg
import dbldatagen.distributions as dist
import pyspark.sql.types
from pyspark.sql.functions import row_number, monotonically_increasing_id, dense_rank, col, lit, to_date
import numpy as np
import pandas as pd
from pyspark.sql import Window
import math
from random import random, randint, shuffle
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
import datetime
np.random.seed()


num_rows = randint(200000, 300000)
num_devices = randint(90, 110)
model_ls = ['SkyJet134', 'SkyJet234', 'SkyJet334', 'EcoJet1000', 'JetLift7000',
            'EcoJet2000', 'FlyForceX550', 'TurboFan3200', 'SkyBolt1', 'SkyBolt2', 
            'MightyWing1100', 'EcoJet3000', 'AeroGlider4150', 'SkyBolt250']
factory_ls = ['A06', 'D18', 'J15', 'C04', 'T10']
amplitude=1.2
frequency=0.35

def sinus_signal(frequency, amplitude, time):
    return np.sin(2*np.pi*frequency * time) * amplitude

def periodic_signal(frequency, amplitude, time):
    freq_val = np.random.normal(loc=frequency, scale=0.5, size=1)
    amplitude_val = np.random.normal(loc=amplitude, scale=1.2, size=1)
    return float(amplitude_val * np.sin(freq_val * time))

def make_timeseries(pattern, num_points, frequency, amplitude, trend_factor=0, noisy=0.3):
    times = np.linspace(0, 10, num_points)
    noise = np.random.normal(loc=0, scale=noisy, size=num_points)
    trend = times * trend_factor

    timeseries = np.zeros(num_points)
    if num_points < 2: 
        return timeseries
    if pattern=='sinusoid':
        for i in range(num_points):
            timeseries[i] = noise[i] + sinus_signal(frequency, amplitude, times[i]) + trend[i]
    elif pattern=='periodic':
        for i in range(num_points):
            timeseries[i] = noise[i] + periodic_signal(frequency, amplitude, times[i]) + trend[i]
    elif pattern=='autoregressive':
        timeseries[0] = random()+2
        timeseries[1] = (random()+1)*2.5
        for i in range(2, num_points):
            timeseries[i] = noise[i] + (timeseries[i-1] * 1.5) - (timeseries[i-2] * .5)
    return timeseries 

def temperature_generator(spark):
    w = Window().orderBy(lit('A')) # TODO: make this an applyinpandas function that returns an autoregressive temperature per day, with pressure
    temp_first = make_timeseries('periodic', num_rows//2, frequency, amplitude, trend_factor=5.9, noisy=7)+25
    temp_second = make_timeseries('periodic', num_rows//2 + num_rows%2, frequency, amplitude, trend_factor=-5.6, noisy=6.5) + temp_first[-1]
    temperature = np.concatenate((temp_first, temp_second))
    df = spark.createDataFrame(temperature).withColumnRenamed('value', 'temperature')
    return df.withColumn('unique_id', row_number().over(w))

def get_datetime_list(length=None):
    if not length:
        length = randint(10, 300)
    start_date = datetime.datetime(2023, 1, 1, 0, 0, 0)
    end_date = datetime.datetime(2023, 12, 31, 23, 59, 59)
    time_diff = (end_date - start_date).total_seconds()
    random_second = randint(0, int(time_diff))
    rand_datetime = start_date + datetime.timedelta(seconds=random_second)
    return pd.date_range(start=str(rand_datetime), periods=length, freq='1 min')

def timestamp_sequence_lengths(total):
    nums = []
    while total > 0:
        if total < 300:
            n = total
        else:
            n = randint(10, 300)
        nums.append(n)
        total -= n
    shuffle(nums)
    return nums

def add_timestamps(pdf: pd.DataFrame) -> pd.DataFrame:
    num_rows = len(pdf)
    lengths = timestamp_sequence_lengths(num_rows)
    timestamp_sequences = [pd.Series(get_datetime_list(length)) for length in lengths]
    pdf['timestamp'] = pd.concat(timestamp_sequences, ignore_index=True)
    return pdf

timestamp_schema = '''device_id string, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
                     pressure double'''

def add_lifetime_features(pdf: pd.DataFrame) -> pd.DataFrame:
    num_points = len(pdf)
    sinusoidal = make_timeseries('sinusoid', num_points, frequency, amplitude, trend_factor=.4, noisy=.6)
    pdf['density'] = abs(sinusoidal - np.mean(sinusoidal)/2) 
    return pdf

lifetime_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
                     pressure double, temperature double, density float'''

def add_trip_features(pdf: pd.DataFrame) -> pd.DataFrame:
    num_points = len(pdf) # TODO: make airflow_rate dependent on rotation_speed
    rotation = abs(make_timeseries('autoregressive', num_points, frequency, amplitude)) *100
    init_delay = make_timeseries('sinusoid', num_points, frequency, amplitude, trend_factor=.2, noisy=1.2)
    pdf['delay'] =  abs(init_delay * np.sqrt(rotation))
    pdf['rotation_speed'] = rotation
    return pdf

trip_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
            rotation_speed double, pressure double, temperature double, delay float, density float'''

def add_defects(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values(['timestamp'])
    pdf['temp_ewma'] = pdf['temperature'].shift(1).ewm(5).mean()
    pdf['temp_difference'] = pdf['temperature'] - pdf['temp_ewma']

    conditions = [
      (pdf['temp_difference'] > 9) & (pdf['model_id'] == 'SkyJet234') & (pdf['temperature'] > 90),
      (pdf['temp_difference'] > 7.9) & (pdf['model_id'] == 'SkyJet334') & (pdf['temperature'] > 90),
      (pdf['delay'] > 105) & (pdf['rotation_speed'] > 1870),
      (pdf['density'] > 4.2) & (pdf['pressure'] < 330),
    ]
    outcomes = [round(random()+.3), round(random()+.3), round(random()+.2), round(random()+.15)]
    pdf['defect'] = np.select(conditions, outcomes, default=0)
    pdf = pdf.drop(['temp_difference', 'temp_ewma'], axis=1)        
    return pdf

defect_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
            rotation_speed double, pressure double, temperature double, delay float, density float, defect float'''


def iot_generator(spark, num_rows=num_rows, num_devices=num_devices):
    # TODO: remove dependency on dbldatagen
    temperature_df = temperature_generator(spark)
    w = Window().orderBy(lit('A'))
    return (
        dg.DataGenerator(sparkSession=spark, name='synthetic_data', rows=num_rows, random=True, randomSeed=num_rows)
        .withColumn('device_id', 'bigint', minValue=1, maxValue=num_devices, random=True)
        .withColumn('model_id', 'string', values=model_ls, base_column='device_id')
        .withColumn('factory_id', 'string', values=factory_ls, base_column='device_id')
        .withColumn('pressure', 'double', minValue=323.0, maxValue=433.53, step=0.01, distribution=dist.Gamma(1.0, 2.0))
        .withColumn('airflow_rate', 'double', minValue=8.0, maxValue=13.3, step=0.1, distribution=dist.Beta(5,1))
        .build()
        .withColumn('device_id', col('device_id').cast('string'))
        .groupBy('device_id').applyInPandas(add_timestamps, timestamp_schema)
        .orderBy('timestamp').withColumn('unique_id', row_number().over(w))
        .join(temperature_df, 'unique_id', 'left').drop('unique_id')
        .withColumn('trip_id', dense_rank().over(Window.partitionBy('device_id').orderBy(to_date('timestamp'))))
        .groupBy('device_id').applyInPandas(add_lifetime_features, lifetime_schema)
        .groupBy('device_id', 'trip_id').applyInPandas(add_trip_features, trip_schema)
        .groupBy('device_id').applyInPandas(add_defects, defect_schema)
    )
