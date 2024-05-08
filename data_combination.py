import pandas as pd
import numpy as np
import os
import sys
import pickle
from utils.utils import load_config



def multicollinearity_search():
    pass

def yield_feature_selection():
    pass


def main():
    print('Combining multi-source datasets to create combined, final datasets for training...')

    config = load_config()
    start_date = config.get('MetaData', 'final_start_date')


    # Download price data (.csv file)
    price_data = pd.read_csv('data/price_data.csv', index_col='Date')
    price_data.index = pd.to_datetime(price_data.index)
    data_m = price_data.resample('ME').last()
    rets_m = np.log(data_m).diff()
    rets_m = rets_m.dropna()
    rets_m_long = pd.DataFrame(rets_m.stack())
    rets_m_long = rets_m_long.reset_index()
    rets_m_long = rets_m_long.rename(columns={0: 'ret', 'level_1': 'Ticker'})

    # Download yield curves (.csv file)
    yield_curve = pd.read_csv('data/interpolated_yield_curves.zip',
        compression='zip',
        index_col='Date'
    )
    # Fix redundant decimals on yield curve names
    yield_curve.columns = [i/10 for i in range(0, 301)]
    yield_curve.index = pd.to_datetime(yield_curve.index)
    yield_curve = yield_curve[[1.0, 10.0, 20.0]]
    yield_curve = yield_curve.rename(columns={1.0: '1 Yr', 10.0: '10 Yr', 20.0: '20 Yr'})
    yield_m = yield_curve.resample('ME').last()


    # Download statement data
    monthly_ff = pd.read_csv('data/monthly_fund_data.csv')
    monthly_ff.Date = pd.to_datetime(monthly_ff.Date)
    monthly_ff.set_index(['Date', 'Ticker'], inplace=True)



    # merge to final dataset
    dataset = rets_m_long.merge(monthly_ff, on=['Date', 'Ticker']).merge(yield_m, on='Date')
    dataset = dataset[dataset.Date >= start_date]


    # pivot to flattened column structure to serve as feed to neural networks
    df = dataset.set_index(['Date', 'Ticker'])
    df_pivot = df.pivot_table(index='Date', columns='Ticker')
    df_pivot.columns = ['_'.join(col) for col in df_pivot.columns]
    
    # send needed datasets to disk
    dataset.to_csv('data/rl_dataset.csv', index=False)
    df_pivot.to_csv('data/final_dataset.csv')

    print('\n\nDatasets are ready for training!\n')


if __name__=="__main__":
    sys.exit(main())