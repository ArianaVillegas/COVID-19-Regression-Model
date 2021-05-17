import pandas as pd
import datetime as dt

from tools import split_data
from test import test

filename = "data/covid.csv"
columns = ['date', 'region', 'lon', 'lat', 'zone', 'confirmed', 'deaths']
covid = pd.read_csv(filename)[columns]

covid.dropna(inplace=True)

# Cambiar el departamento aqui
# covid = covid[covid['region'] == 'LIMA']

covid.loc[:, 'date_num'] = pd.to_datetime(covid['date'])
covid.loc[:, 'date_num'] = covid['date_num'].map(dt.datetime.toordinal)

# covid_region, regions = split_data(covid, 'region')
covid_region, regions = [covid], ['PERU']
labels = ['confirmed', 'deaths']

train_size = 0.7
alpha_values = [0.001]
iterations_values = [100, 500, 1000, 5000, 10000]
error_methods = ['mean_absolute_error', 'mean_squared_error']
regularization = ['l1', 'l2']
opt_methods = ['momentum', 'adagrad', 'adadelta', 'adam']

path = ['output_3v_nonlinear']

test(path, covid_region, regions, labels, alpha_values, iterations_values, error_methods, regularization,
     opt_methods, train_size)
