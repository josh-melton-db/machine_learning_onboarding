from pyspark.sql.functions import (
    rand, expr, row_number, monotonically_increasing_id, 
    dense_rank, col, lit, to_date
)
from pyspark.sql import Window
import pandas as pd
import numpy as np
np.random.seed(0)
import datetime
import random
random.seed(0)
from random import randint, shuffle, random


num_rows = 250000
num_devices = 200
amplitude = 1.2
frequency = 0.35

def create_initial_df(spark, num_rows, num_devices):
    factory_ls = ["'A06'", "'D18'", "'J15'", "'C04'", "'T10'"]
    model_ls = ["'SkyJet134'", "'SkyJet234'", "'SkyJet334'", "'EcoJet1000'", "'JetLift7000'",
                "'EcoJet2000'", "'FlyForceX550'", "'TurboFan3200'", "'SkyBolt1'", "'SkyBolt2'",
                "'MightyWing1100'", "'EcoJet3000'", "'AeroGlider4150'", "'SkyBolt250'"]
    return (
        spark.range(num_rows).withColumn('device_id', (rand(seed=0)*num_devices).cast('int') + 1)
        .withColumn('factory_id', expr(f"element_at(array({','.join(factory_ls)}), abs(hash(device_id)%{len(factory_ls)})+1)"))
        .withColumn('model_id', expr(f"element_at(array({','.join(model_ls)}), abs(hash(device_id)%{len(model_ls)})+1)"))
        .drop('id')
    )

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

timestamp_schema = '''device_id string, factory_id string, model_id string, timestamp timestamp'''

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
        timeseries[0] = random() + noisy
        timeseries[1] = (random()+1) * (noisy + .5)
        for i in range(2, num_points):
            timeseries[i] = noise[i] + (timeseries[i-1] * (1+trend_factor)) - (timeseries[i-2] * (1-trend_factor))
    return timeseries 

def get_starting_temps(noisy=.3, trend=.1, mean=58, std_dev=17) -> pd.DataFrame:
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    num_rows = len(dates)
    normal1 = np.random.normal(loc=mean, scale=std_dev, size=num_rows // 2).clip(min=0, max=78)
    normal1 = np.sort(normal1)
    normal2 = np.random.normal(loc=mean, scale=std_dev, size=num_rows // 2 + num_rows % 2).clip(min=0, max=78)
    normal2 = np.sort(normal2)[::-1]
    normal = np.concatenate((normal1, normal2))
    noise = 9.5 * make_timeseries('autoregressive', num_rows, frequency, amplitude, noisy=noisy, trend_factor=trend)
    temps = normal + noise
    pdf = pd.DataFrame({'date': dates, 'starting_temp': temps})
    pdf['date'] = pdf['date'].dt.date
    return pdf

def add_weather(pdf: pd.DataFrame) -> pd.DataFrame:
    num_rows = len(pdf)
    start_temp = pdf['starting_temp'].iloc[0]
    pdf['pd_timestamp'] = pd.to_datetime(pdf['timestamp'])
    min_time = pdf['pd_timestamp'].min()
    coldest = pdf['pd_timestamp'].dt.normalize() + pd.DateOffset(hours=randint(5, 8))
    hottest = pdf['pd_timestamp'].dt.normalize() + pd.DateOffset(hours=randint(14, 18))
    hottest_time = hottest[0]
    coldest_time = coldest[0]
    timedelta_from_hottest = hottest - pdf['pd_timestamp']
    peak_timestamp_idx = min(num_rows-1, timedelta_from_hottest.idxmin())
    upwards = min_time < hottest_time and min_time > coldest_time
    if upwards:
        trend_factor = .8
    else:
        trend_factor = -.8
    pdf['temperature'] = start_temp + make_timeseries('periodic', num_rows, frequency, amplitude, trend_factor=-.8, noisy=1)
    pdf = pdf.drop('pd_timestamp', axis=1)
    random_lapse_rate = 1.2 + np.random.uniform(.5, 1.5)
    pdf['air_pressure'] = randint(913, 1113) - (pdf['temperature'] - 15) * random_lapse_rate
    return pdf

weather_schema = '''device_id string, factory_id string, model_id string, timestamp timestamp, date date,
                    trip_id integer, starting_temp double, temperature double, air_pressure double'''

def add_lifetime_features(pdf: pd.DataFrame) -> pd.DataFrame:
    num_points = len(pdf)
    sinusoidal = make_timeseries('sinusoid', num_points, frequency, amplitude, trend_factor=.4, noisy=.6)
    pdf['density'] = abs(sinusoidal - np.mean(sinusoidal)/2) 
    return pdf

lifetime_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp,  
                     air_pressure double, temperature double, density float'''

def add_trip_features(pdf: pd.DataFrame) -> pd.DataFrame:
    num_points = len(pdf)
    rotation = abs(make_timeseries('autoregressive', num_points, frequency, amplitude)) *100
    init_delay = make_timeseries('sinusoid', num_points, frequency, amplitude, trend_factor=.2, noisy=1.2)
    pdf['delay'] =  abs(init_delay * np.sqrt(rotation))
    pdf['rotation_speed'] = rotation
    pdf['airflow_rate'] =  pdf['rotation_speed'].shift(5) / pdf['air_pressure']
    pdf = pdf.fillna(method='bfill') # TODO: fix the missing temperatures more effectively
    pdf = pdf.fillna(method='ffill')
    pdf = pdf.fillna(0)
    return pdf

trip_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
            rotation_speed double, air_pressure double, temperature double, delay float, density float'''

def add_defects(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf = pdf.sort_values(['timestamp'])
    pdf['temp_ewma'] = pdf['temperature'].shift(1).ewm(5).mean()
    pdf['temp_difference'] = pdf['temperature'] - pdf['temp_ewma']

    conditions = [ # TODO: don't hardcode temps, take value at some percentile to help with different dataset sizes
      (pdf['temp_difference'] > 1.5) & (pdf['model_id'] == 'SkyJet234') & (pdf['temperature'] > 84), 
      (pdf['temp_difference'] > 1.3) & (pdf['model_id'] == 'SkyJet334') & (pdf['temperature'] > 87),
      (pdf['delay'] > 40) & (pdf['rotation_speed'] > 590),
      (pdf['density'] > 4.3) & (pdf['air_pressure'] < 780), # TODO: add in some factory_id dependence as well
    ]
    outcomes = [round(random()+.3), round(random()+.3), round(random()+.2), round(random()+.15)]
    pdf['defect'] = np.select(conditions, outcomes, default=0)
    pdf = pdf.drop(['temp_difference', 'temp_ewma'], axis=1)        
    return pdf

defect_schema = '''device_id string, trip_id int, factory_id string, model_id string, timestamp timestamp, airflow_rate double,  
                    rotation_speed double, air_pressure double, temperature double, delay float, density float, defect float'''

def generate_iot(spark, num_rows=num_rows, num_devices=num_devices):
    starting_temps = spark.createDataFrame(get_starting_temps())
    return (
        create_initial_df(spark, num_rows, num_devices)
        .withColumn('device_id', col('device_id').cast('string'))
        .groupBy('device_id').applyInPandas(add_timestamps, timestamp_schema)
        .withColumn('date', to_date(col('timestamp')))
        .withColumn('trip_id', dense_rank().over(Window.partitionBy('device_id').orderBy('date')))
        .join(starting_temps, 'date', 'left')
        .groupBy('trip_id', 'device_id').applyInPandas(add_weather, weather_schema)
        .drop('starting_temp', 'date')
        .groupBy('device_id').applyInPandas(add_lifetime_features, lifetime_schema)
        .groupBy('device_id', 'trip_id').applyInPandas(add_trip_features, trip_schema)
        .groupBy('device_id').applyInPandas(add_defects, defect_schema)
    )