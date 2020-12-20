import pickle
from pandas_datareader.data import DataReader
import pandas as pd
import numpy as np
import scipy.stats as stats

def load_file(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_file(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)


def get_data_for_multiple_stocks(tickers, start_date, end_date):
    '''
    tickers: list of tickers to get data for
    start_date, end_date: dt.datetime objects
    method returns a dictionary b{ticker: pd.DataFrame}
    '''
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    stocks = dict()
    # loop through all the tickers
    for i, ticker in enumerate(tickers):
        if i % 5 == 0:
            print(f'{i}/{len(tickers)}')

        try:
            # get the data for the specific ticker
            s = DataReader(ticker, 'yahoo', start_date_str, end_date_str)

            s.insert(0, "Ticker", ticker)

            s['Prev Close'] = s['Adj Close'].shift(1)
            s['daily_return'] = (s['Adj Close']/s['Prev Close']) - 1
            s['log_return'] = np.log(s['Adj Close']/s['Prev Close'])
            # s['perc_return'] = (s['Adj Close']/s['Prev Close'])
            # add it to the dictionary
            stocks[ticker] = s
        except:
            print(f'something went wrong with {ticker}')
            continue

    # return the dictionary
    return stocks


def get_df_by_metric(data_dict, metric):
    '''
    data_dict: dictionary of the indiv ticker data

    metric: the metric of the data you want (eg. Adj Close)

    returns pd.DataFrame with each column representing one ticker
    '''
    output = pd.DataFrame()

    for k, v in data_dict.items():
        output[k] = v[metric]

    output.dropna(how='all', inplace=True)
    return output


def rebalance_weights(weights, delisted_ticker):
    '''
    

    Parameters
    ----------
    weights : pd.Series
        weights of assets in portfolio. Series indexed by asset ticker
        
    delisted_ticker : str
        ticker of delisted asset. Weight will be set to 0

    Returns
    -------
        rebalanced_weights

    '''
    if delisted_ticker is None:
        return weights
    
    else:   
        rebalanced_weights = weights.copy(deep=True)
        rebalanced_weights.loc[delisted_ticker] = 0
        rebalanced_weights = rebalanced_weights / rebalanced_weights.sum()
        assert np.isclose(rebalanced_weights.sum(), 1, atol=1e-3)
        return rebalanced_weights
    

def evaluate_portfolio(log_returns_df, weights, delisted_assets={}):
    
    '''
    

    Parameters
    ----------
    
    delisted_assets : dictionary{ticker: delist date}, optional
        asset ticker passed as string
        delist date passed as dt.Date object
        if not None, returns will be calculated after rebalancing weights on delist date
        
    Returns
    -------
        portfolio risk and return

    '''    
    # keep track of start and end date
    curr_start = log_returns_df.index[0]
    curr_end = log_returns_df.index[-1]
    
    # to make sure returns series extends all the way to end
    delisted_assets[None] = log_returns_df.index[-1]

    # init variables
    overall_returns = pd.Series()
    curr_weights = weights

    for ticker, delist_date in sorted(delisted_assets.items(), key=lambda x: x[1]):
        curr_end = delist_date

        period_log_returns_df = log_returns_df[(log_returns_df.index >= pd.to_datetime(curr_start))
                                               & (log_returns_df.index <= pd.to_datetime(curr_end))]

        period_returns = np.dot(curr_weights, np.exp(period_log_returns_df).T - 1)
        period_returns = pd.Series(period_returns, index=period_log_returns_df.index)

        # update start and end
        delist_date_idx = log_returns_df.index.get_loc(delist_date)
        curr_start = log_returns_df.index[min(delist_date_idx, log_returns_df.shape[0])]

        # update weights
        curr_weights = rebalance_weights(curr_weights, ticker)

        # update series
        overall_returns = overall_returns.append(period_returns)
  
    # get monthly returns and risk
    monthly_returns = overall_returns.groupby(pd.Grouper(freq='M')).apply(lambda x: (x+1).prod() - 1)
    sd = overall_returns.groupby(pd.Grouper(freq='M')).apply(lambda x: np.std(x))
    monthly_annualised_sd = np.sqrt(252) * sd

    return monthly_returns, monthly_annualised_sd

def get_beta(index_series, asset_series, rf_series, return_r2=False):
    index_excess_returns = index_series.subtract(rf_series / 252)
    asset_excess_returns = asset_series.subtract(rf_series / 252)
    
    # linear regression
    result = stats.linregress(x=index_excess_returns, y=asset_excess_returns)
    beta = result[0]
    r2 = result[2]

    if return_r2:
        return beta, r2
    return beta

def get_monthly_alpha(index_series, asset_series, rf_series):
    '''
    all series should be indexed by datetime
    returns the alpha for each month
    if no data available for asset for given month, returns NaN
    '''

    index_excess_returns = index_series.subtract(rf_series / 252)
    asset_excess_returns = asset_series.subtract(rf_series / 252)

    df = pd.concat([index_excess_returns, asset_excess_returns], axis=1)
    df.columns = ['index', 'asset']

    # group by month
    alphas = df.groupby(pd.Grouper(freq='M')).apply(lambda group: stats.linregress(x=group['index'], y=group['asset'])[1])

    return alphas

    
    
    