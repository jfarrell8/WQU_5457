import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, SimpleRNN, Dense, Dropout
from utils.utils import set_seeds, load_config, plot_wealth_index
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
import datetime
import time
import ast
import sys

class PortfolioModelTrainer:
    def __init__(self, model_type, asset_num, input_shape, param_grid, wts=None):
        self.model_type = model_type
        self.model = None
        self.asset_num = asset_num
        self.input_shape = input_shape
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = np.inf
        self.wts = wts

    # custom loss function for gradient ascent Sharpe Ratio
    def negative_sharpe_loss(self, wts, rets):
        mean_return = tf.reduce_mean(tf.reduce_sum(wts * rets, axis=1))
        std_return = tf.math.reduce_std(tf.reduce_sum(wts * rets, axis=1))

        return -mean_return / std_return

    def build_model(self, params):
        self.model = Sequential()

        if self.model_type == "CNN":

          self.model.add(Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], activation=params['activation'], input_shape=self.input_shape))
          self.model.add(MaxPooling1D(pool_size=2))
          self.model.add(Flatten())
          self.model.add(Dropout(params['dropout_rate']))
          self.model.add(Dense(params['dense_units'], activation=params['activation']))
        
        elif self.model_type == "RNN":
           self.model.add(SimpleRNN(params['input_layer_size'], input_shape=self.input_shape))
           self.model.add(Dropout(params['dropout_rate']))
        
        elif self.model_type == "MLP":

          for i in range(params['num_hidden_layers']):
            if i==0:
              self.model.add(Dense(units=params['hidden_layer_sizes'], activation=params['activation'], input_shape=self.input_shape))
            else:
              self.model.add(Dense(units=params['hidden_layer_sizes'], activation=params['activation']))

            if params['dropout_rate'] > 0.0:
              self.model.add(Dropout(params['dropout_rate']))

        else:
           raise ValueError("Invalid model type. Supported types: 'CNN', 'RNN', 'MLP'")
        
        self.model.add(Dense(units=self.asset_num, activation="softmax")) # <- softmax creates long-only portfolios
        
        optimizer_instance = tf.keras.optimizers.get(params['optimizer'])
        optimizer_instance.learning_rate = params['learning_rate']

        self.model.compile(optimizer=optimizer_instance, loss=self.negative_sharpe_loss)

        # return self.model

    def walk_forward_train(self, n_train, n_val, df, batch, params):
        n_splits = (df.shape[0] - n_train) // n_val + 1
        avg_score = 0.0

        preds = []

        for i in range(0, df.shape[0] - n_train, n_val):

            X_train, y_train = df.iloc[i : i + n_train, :-self.asset_num], df.iloc[i : i + n_train, -self.asset_num:]
            X_val, y_val = df.iloc[i + n_train : i + n_train + n_val, :-self.asset_num], df.iloc[i + n_train : i + n_train + n_val, -self.asset_num:]

            # Define EarlyStopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            history = self.model.fit(
                X_train,
                y_train,
                epochs=params['epochs'],
                batch_size=batch,
                validation_data = (X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )

            y_pred = self.model.predict(X_val, verbose=0)
            preds.append(y_pred)

            score = self.model.evaluate(X_val, y_val, verbose=0)
            avg_score += score / n_splits

        return preds, avg_score
    
    def grid_search(self, n_train, n_val, df, batch):
       total_combos = len(ParameterGrid(self.param_grid))

       for params in ParameterGrid(self.param_grid):
            print(f'Only {total_combos} left!!')
            print('Testing parameters: ', params)
            time.sleep(1)
          
            self.build_model(params)
          
            preds, avg_score = self.walk_forward_train(n_train=n_train,
                                                      n_val=n_val,
                                                      df=df,
                                                      batch=batch,
                                                      params=params
                                                )

            if avg_score < self.best_score:
                self.best_score = avg_score
                self.best_params = params

            total_combos -= 1
          


    def train_optimal(self, df, batch):

        # train with best params
        self.build_model(self.best_params)

        history = self.model.fit(df.iloc[:, :-self.asset_num],
                                        df.iloc[:, -self.asset_num:],
                                        epochs=self.best_params['epochs'],
                                        batch_size=batch,
                                        verbose=1)


    def evaluate(self, df, date_index, tickers):
        y_pred = self.model.predict(df.iloc[:, :-self.asset_num])
        self.wts = pd.DataFrame(y_pred, index=date_index, columns=tickers)

        return self.wts
    
    def save_model(self, timestamp, model_version):

        self.model.save(f"../data/models/{self.model_type}_{timestamp}")

        self.wts.to_csv(f"../data/model_data/{self.model_type}_{model_version}_{timestamp}_wts.csv")

    def build_benchmark(self, date_index, tickers):
        if self.model_type == 'EW':
            bench_wts = pd.DataFrame(index = date_index, columns = tickers)
            bench_wts[:] = 1 / len(bench_wts.columns)

        self.wts = bench_wts

        # return bench_wts
    
    def perform_backtest(self, price_data):

        # portfolio value over time
        portfolio_value = (price_data * self.wts).sum(axis=1)

        # monthly portfolio returns
        portfolio_returns = portfolio_value.pct_change()
        p_rets = pd.DataFrame({self.model_type:portfolio_returns})


        # total return, annualized return, Sharpe ratio, Volatility, Max Drawdown
        totalReturn = portfolio_value[-1] / portfolio_value[0] - 1
        years = len(portfolio_value) / 12
        annualReturn = ((portfolio_value[-1] / portfolio_value[0]) ** (1 / years)) - 1
        monthly_returns = portfolio_value.pct_change()
        sharpeRatio = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
        volatility = monthly_returns.std() * np.sqrt(12)
        cumulative_returns = (1 + monthly_returns).cumprod()
        maxDrawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

        p_metrics = pd.DataFrame({f'{self.model_type}':[totalReturn, annualReturn, sharpeRatio, volatility, maxDrawdown]})
        p_metrics.index = ['totalReturn', 'annualReturn', 'sharpeRatio', 'volatility', 'maxDrawdown']

        return p_metrics, p_rets

def config_dict_parser(config, section_name):
    section_dict = {}
    for key, value in config[section_name].items():
        try:
            # Attempt to parse the value as a list
            section_dict[key] = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            # If parsing as a list fails, keep the value as a string
            section_dict[key] = value

    return section_dict


def main():

    # set seeds
    set_seeds(seed_state=42)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # load config to get params
    config = load_config()
    trainval_split = float(config.get('NeuralNetParams', 'trainval_split'))
    asset_num = int(config.get('MetaData', 'ticker_num'))
    n_train = int(config.get('NeuralNetParams', 'n_train'))
    n_val = int(config.get('NeuralNetParams', 'n_val'))
    batch = int(config.get('NeuralNetParams', 'batch'))
    initial_amount = int(config.get('MetaData', 'initial_amount'))

    # load master dataset
    df_pivot = pd.read_csv('../data/final_dataset.csv', index_col='Date')
    df_pivot.index = pd.to_datetime(df_pivot.index)
    ticker_list = [i.split('ret_')[1] for i in df_pivot.columns[-asset_num:]]

    # let's establish train, validation, and test periods here:
    trainval_date_index = df_pivot.index[:int(trainval_split*df_pivot.shape[0])]
    test_date_index = df_pivot.index[int(trainval_split*df_pivot.shape[0]):]

    # load and prep price data for performance metric calcs later
    price_data = pd.read_csv('../data/price_data.csv', index_col='Date')
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.resample('ME').last()

    test_prices = price_data[price_data.index.isin(test_date_index)]
    test_prices = test_prices.sort_index()

    train_prices = price_data[price_data.index.isin(trainval_date_index)]
    train_prices = train_prices.sort_index()

    # price_data = price_data[price_data.index.isin(test_date_index)]
    # price_data = price_data.sort_index()

    # need to scale our full dataset for training
    df_pivot.reset_index(inplace=True, drop=True)

    # let's re-scale all of the data points
    scaler = MinMaxScaler()
    df_pivot_scaled = pd.DataFrame(scaler.fit_transform(df_pivot))

    # now set test dataset aside for post-training evaluation
    df_trainval = df_pivot_scaled.iloc[:int(trainval_split*df_pivot.shape[0])]
    df_test = df_pivot_scaled.iloc[int(trainval_split*df_pivot.shape[0]):]
    
    train_metrics = pd.DataFrame()
    test_metrics = pd.DataFrame()
    train_rets = pd.DataFrame()
    test_rets = pd.DataFrame()
    
    for model_type in ['MLP', 'CNN', 'RNN']:
        print(f'Training: {model_type}...')

        if model_type == 'MLP':
            input_shape = df_trainval.iloc[:, :-asset_num].shape[1]
            input_shape = (input_shape,)
        elif model_type == 'CNN' or model_type == 'RNN':
            # need to reshape for CNNs and RNNs
            X_train_full = df_trainval.iloc[:, :-asset_num]
            X_train_reshaped = np.reshape(X_train_full.to_numpy(), (X_train_full.shape[0], X_train_full.shape[1], 1))
            input_shape = X_train_reshaped.shape[1:]
        
        # param_grid = config.get('GridSearchParams', f'{model_type.lower()}_param_grid')
        param_grid = config_dict_parser(config, f'{model_type.lower()}_param_grid')

        # instantiate the portfolio object
        portfolio = PortfolioModelTrainer(model_type=model_type, 
                                          asset_num=asset_num,
                                          input_shape=input_shape, 
                                          param_grid=param_grid)
        
        # perform grid_search on the dataset
        portfolio.grid_search(n_train, n_val, df_trainval, batch)

        print(f'Found best params for {model_type}!')
        print(f'Training optimal model for {model_type}...')

        # train optimal
        portfolio.train_optimal(df_trainval, batch)

        # let's see how it performs in-sample
        train_wts = portfolio.evaluate(df_trainval, trainval_date_index, ticker_list)

        # perform in-sample backtest
        p_train_metrics, p_train_rets = portfolio.perform_backtest(train_prices)

        # add to total_metrics and total_rets for later analysis
        train_metrics = pd.concat([train_metrics, p_train_metrics], axis=1)
        train_rets = pd.concat([train_rets, p_train_rets], axis=1)

        # save train preds
        portfolio.save_model(timestamp, 'train')

        # predict out-of-sample performance on test dataset
        test_wts = portfolio.evaluate(df_test, test_date_index, ticker_list)

        # save model and test preds
        portfolio.save_model(timestamp, 'test')

        # perform backtest on out-of-sample test predictions
        p_test_metrics, p_test_rets = portfolio.perform_backtest(test_prices)

        # add to total_metrics and total_rets for later analysis
        test_metrics = pd.concat([test_metrics, p_test_metrics], axis=1)
        test_rets = pd.concat([test_rets, p_test_rets], axis=1)

        print(f'Done training {model_type}!!!\n\n\n')


    # let's capture performance for both train and test samples
    for model_version in ['train', 'test']:
        if model_version == 'train':
           date_index = trainval_date_index
           prices = train_prices
           metrics = train_metrics
           rets = train_rets
        elif model_version == 'test':
           date_index = test_date_index
           prices = test_prices
           metrics = test_metrics
           rets = test_rets

        # get benchmark equal-weighted portfolio performance
        ew_portfolio = PortfolioModelTrainer(model_type='EW',
                                            asset_num=asset_num,
                                            input_shape=None,
                                            param_grid=None)
        # ew train performance
        ew_portfolio.build_benchmark(date_index, ticker_list)
        ew_metrics, ew_rets = ew_portfolio.perform_backtest(prices)

        metrics = pd.concat([metrics, ew_metrics], axis=1)
        rets = pd.concat([rets, ew_rets], axis=1)

        # save metrics and returns to disk
        metrics.to_csv(f'../data/model_data/{model_version}_metrics_{timestamp}.csv')
        rets.to_csv(f'../data/model_data/{model_version}_rets_{timestamp}.csv')

        # plot and save wealth index
        plot_wealth_index(rets, timestamp, initial_amount, model_version)


if __name__=="__main__":
   sys.exit(main())