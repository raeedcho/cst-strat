# This script runs a single neuron comparison between center-out and CST

# %% imports
import numpy as np
import pandas as pd
import cst
import pyaldata
import matplotlib as mpl
import matplotlib.pyplot as plt

# %% load data
file_query = {
    'monkey': 'Earl',
    'session_date': '20190716'
}
td = cst.load_clean_data(file_query)

# %%
avg_fr_table = cst.single_neuron_analysis.get_task_epoch_neural_averages(td)

# %%
