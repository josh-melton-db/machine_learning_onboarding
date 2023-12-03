# Databricks notebook source
pip install -q git+https://github.com/TimeSynth/TimeSynth.git

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"
import timesynth as ts
import pandas as pd
np.random.seed()

# COMMAND ----------


