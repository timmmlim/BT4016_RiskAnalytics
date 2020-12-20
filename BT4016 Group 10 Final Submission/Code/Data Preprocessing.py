from Utils import *
import pandas as pd
from pathlib import Path
import datetime as dt
from os.path import join

# %% download and save data
preload = True
data_path = Path(r'..\Midterm Report\data')
biotech_stocks = pd.read_csv(join(data_path, 'biotech_stocks.csv'))

tickers = biotech_stocks['Symbol']

start_date = dt.date(2019, 1, 1)
end_date = dt.date(2020, 10, 30)

if preload:
    data = load_file(join(data_path, 'xgboost_data.csv'))
else:
    data = get_data_for_multiple_stocks(tickers, start_date, end_date)
    save_file(join(data_path, 'xgboost_data.csv'), data)

# %% get returns data
index = get_data_for_multiple_stocks(['^NBI'], start_date, end_date)
index_returns = get_df_by_metric(index, 'log_return')
data_returns = get_df_by_metric(data, 'log_return')

# %% rf rate 2019 and 2020o
# 2019
rf_2019 = pd.read_csv(join(data_path, 'yield_rate_2019.csv'))
rf_2019.drop(rf_2019.columns.difference(['Date','1 Yr']), 1, inplace=True)

rf_2019['1 Yr'] = rf_2019['1 Yr'].apply(lambda x: x / 100)
rf_2019['Date'] = pd.to_datetime(rf_2019['Date'])
rf_2019.set_index("Date", inplace=True)
rf_2019.rename(columns={'1 Yr': 'risk_free_rate'}, inplace=True)

# 2020
rf_2020 = pd.read_csv(join(data_path, 'yield_rate_2020.csv'))
rf_2020.drop(rf_2020.columns.difference(['Date','1 Yr']), 1, inplace=True)

rf_2020['1 Yr'] = rf_2020['1 Yr'].apply(lambda x: x / 100)
rf_2020['Date'] = pd.to_datetime(rf_2020['Date'])
rf_2020.set_index("Date", inplace=True)
rf_2020.rename(columns={'1 Yr': 'risk_free_rate'}, inplace=True)

# combine series and align index
rf = rf_2019.append(rf_2020)
rf = index_returns.merge(rf, how='left', left_index=True, right_index=True)['risk_free_rate']
rf.fillna(method='ffill', inplace=True)
# %% calculate monthly alpha
alphas = pd.DataFrame()

for i, ticker in enumerate(data_returns.columns):
    if i % 5 == 0:
        print (f'{i}/{data_returns.shape[1]}')

    asset_return_series = data_returns[ticker]
    asset_alphas = get_monthly_alpha(np.exp(index_returns['^NBI']) - 1, 
                                     np.exp(asset_return_series) -1,
                                     rf)
    alphas[ticker] = asset_alphas

alphas.to_csv(join(data_path, 'monthly_alphas.csv'))
    
    
    