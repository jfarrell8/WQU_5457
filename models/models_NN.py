import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from cvxpy import Variable, Minimize, quad_form, Problem
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, SimpleRNN, Dense, Dropout
from utils.utils import set_seeds, load_config, config_dict_parser, plot_wealth_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import datetime
import time
import sys
import os

class PortfolioModelTrainer:
    def __init__(self, model_type, asset_num, input_shape, param_grid, timestamp, wts=None):
        self.model_type = model_type
        self.model = None
        self.asset_num = asset_num
        self.input_shape = input_shape
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = np.inf
        self.wts = wts
        self.timestamp = timestamp

    # custom loss function for gradient ascent Sharpe Ratio
    def negative_sharpe_loss(self, rets, wts):
        mean_return = tf.reduce_mean(tf.reduce_sum(wts * rets, axis=1))
        std_return = tf.math.reduce_std(tf.reduce_sum(wts * rets, axis=1))
        if std_return != 0.0:
            return -mean_return / std_return
        else:
            return 0.0

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


    def walk_forward_train(self, n_train, n_val, df, batch, params):
        n_splits = (df.shape[0] - n_train) // n_val + 1
        avg_score = 0.0

        preds = []

        for i in range(0, df.shape[0] - n_train, n_val):

            if params['window_type'] == 'fix':
                X_train, y_train = df.iloc[i : i + n_train, :-self.asset_num], df.iloc[i : i + n_train, -self.asset_num:]
                X_val, y_val = df.iloc[i + n_train : i + n_train + n_val, :-self.asset_num], df.iloc[i + n_train : i + n_train + n_val, -self.asset_num:]

            elif params['window_type'] == 'exp':
                X_train, y_train = df.iloc[: i + n_train, :-self.asset_num], df.iloc[: i + n_train, -self.asset_num:]
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

        # if not os.path.exists(f"../data/models/{timestamp}/"):
        #     os.makedirs(f"../data/models/{timestamp}/")
        # if not os.path.exists(f"../data/model_data/{timestamp}/"):
        #     os.makedirs(f"../data/model_data/{timestamp}/")

        self.model.save(f"../data/models/{timestamp}/{self.model_type}")
        self.wts.to_csv(f"../data/model_data/{timestamp}/{self.model_type}/{self.model_type}_{model_version}_wts.csv")

    def build_benchmark(self, date_index, tickers, rets_m=None, n_train=None):
        if self.model_type == 'EW':
            bench_wts = pd.DataFrame(index = date_index, columns = tickers)
            bench_wts[:] = 1 / len(bench_wts.columns)
        elif self.model_type == 'MVO':
            mvo_wts = pd.DataFrame()
            for i in range(2, len(rets_m) - n_train + 1):
                rets_exp = rets_m.iloc[i : i + n_train, :].mean()
                rets_exp = rets_exp.values.reshape(1, -1)
                rets_cov = rets_m.iloc[i : i + n_train, :].cov()

                x = Variable(rets_m.shape[1])

                # return and risk in proper matrix form
                ret = rets_exp @ x
                risk = quad_form(x, rets_cov)

                # optimize with constraints (long-only and all weights sum to 1)
                prob = Problem(Minimize(risk), [sum(x)==1, x >= 0])

                prob.solve()

                weights = pd.DataFrame([x.value], columns=rets_m.columns).round(4).T.rename(columns={0:rets_m.index[i + n_train-1]}).T

                mvo_wts = pd.concat([mvo_wts, weights], axis=0)

            bench_wts = mvo_wts[mvo_wts.index.isin(date_index)]

        self.wts = bench_wts
    
    def perform_backtest(self, price_data, initial_amount):


        ind_rets = price_data.pct_change().dropna()
        portfolio_returns = (ind_rets * self.wts.iloc[1:,:]).sum(axis=1)
        p_rets = pd.DataFrame({self.model_type: portfolio_returns})

        cumulative_rets = (1 + portfolio_returns).cumprod()
        wealth_index = initial_amount * cumulative_rets
        
        sharpeRatio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12)
        totalReturn = wealth_index[-1] / wealth_index[0] - 1
        years = len(portfolio_returns) / 12
        annualReturn = (wealth_index[-1]/wealth_index[0])**(1 / years) - 1
        volatility = portfolio_returns.std() * np.sqrt(12)
        maxDrawdown = (cumulative_rets / cumulative_rets.cummax() - 1).min()


        p_metrics = pd.DataFrame({f'{self.model_type}':[totalReturn, annualReturn, sharpeRatio, volatility, maxDrawdown]})
        p_metrics.index = ['totalReturn', 'annualReturn', 'sharpeRatio', 'volatility', 'maxDrawdown']

        return p_metrics, p_rets

def directory_creator(timestamp, neural_networks):
    # if not os.path.exists(f"../data/model_data/{timestamp}/"):
    #     os.makedirs(f"../data/model_data/{timestamp}/")
    # if not os.path.exists(f"../data/models/{timestamp}/"):
    #     os.makedirs(f"../data/models/{timestamp}/")
    for nn_type in neural_networks:
        if not os.path.exists(f"../data/models/{timestamp}/{nn_type}/"):
            os.makedirs(f"../data/models/{timestamp}/{nn_type}/")
        if not os.path.exists(f"../data/model_data/{timestamp}/{nn_type}/"):
            os.makedirs(f"../data/model_data/{timestamp}/{nn_type}/")

    

def main():

    # set seeds
    set_seeds(seed_state=42)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    neural_networks = ['MLP', 'CNN', 'RNN']
    
    # create directories
    directory_creator(timestamp, neural_networks)

    # load config to get params
    config = load_config()
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
    trainval_date_index = df_pivot.index[:-n_val]
    test_date_index = df_pivot.index[-n_val:]

    # load and prep price data for performance metric calcs later
    price_data = pd.read_csv('../data/price_data.csv', index_col='Date')
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.resample('ME').last()

    test_prices = price_data[price_data.index.isin(test_date_index)]
    test_prices = test_prices.sort_index()

    train_prices = price_data[price_data.index.isin(trainval_date_index)]
    train_prices = train_prices.sort_index()


    # need to scale our full dataset for training
    df_pivot.reset_index(inplace=True, drop=True)

    # let's re-scale all of the data points
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(df_pivot.iloc[:, :-61]))
    unscaled_rets = df_pivot.iloc[:, -61:].reset_index(drop=True)
    scaled_dataset = pd.concat([scaled_features, unscaled_rets], axis=1)

    # now set test dataset aside for post-training evaluation
    df_trainval = scaled_dataset.iloc[:-n_val]
    df_test = scaled_dataset.iloc[-n_val:]
    
    train_metrics = pd.DataFrame()
    test_metrics = pd.DataFrame()
    train_rets = pd.DataFrame()
    test_rets = pd.DataFrame()
    
    for model_type in neural_networks:
        print(f'Training: {model_type}...')

        if model_type == 'MLP':
            input_shape = df_trainval.iloc[:, :-asset_num].shape[1]
            input_shape = (input_shape,)
        elif model_type == 'CNN' or model_type == 'RNN':
            # need to reshape for CNNs and RNNs
            X_train_full = df_trainval.iloc[:, :-asset_num]
            X_train_reshaped = np.reshape(X_train_full.to_numpy(), (X_train_full.shape[0], X_train_full.shape[1], 1))
            input_shape = X_train_reshaped.shape[1:]
        
        param_grid = config_dict_parser(config, f'{model_type.lower()}_param_grid')

        # instantiate the portfolio object
        portfolio = PortfolioModelTrainer(model_type=model_type, 
                                          asset_num=asset_num,
                                          input_shape=input_shape, 
                                          param_grid=param_grid,
                                          timestamp=timestamp)
        
        batch = df_trainval.shape[0]

        # perform grid_search on the dataset
        portfolio.grid_search(n_train, n_val, df_trainval, batch)

        print(f'Found best params for {model_type}!')
        with open(f"../data/model_data/{timestamp}/{portfolio.model_type}/param_mapping.txt", "a") as f:
            f.write(f'The best params for {portfolio.model_type} were: {str(portfolio.best_params)}\n')
        print(f'Training optimal model for {model_type}...')

        # train optimal
        portfolio.train_optimal(df_trainval, batch)

        for model_version in ['train', 'test']:
            if model_version == 'train':
                df = df_trainval
                date_index = trainval_date_index
                prices = train_prices
            elif model_version == 'test':
                df = df_test
                date_index = test_date_index
                prices = test_prices
            pred_wts = portfolio.evaluate(df, date_index, ticker_list)
            metrics, rets = portfolio.perform_backtest(prices, initial_amount)

            if model_version == 'train':
                train_metrics = pd.concat([train_metrics, metrics], axis=1)
                train_rets = pd.concat([train_rets, rets], axis=1)
            elif model_version == 'test':
                test_metrics = pd.concat([test_metrics, metrics], axis=1)
                test_rets = pd.concat([test_rets, rets], axis=1)   

            portfolio.save_model(timestamp, model_version)            
            

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
                                            param_grid=None,
                                            timestamp=timestamp)
        # ew train performance
        ew_portfolio.build_benchmark(date_index, ticker_list)
        ew_metrics, ew_rets = ew_portfolio.perform_backtest(prices, initial_amount)

        # get MVO portfolio performance
        mvo_portfolio = PortfolioModelTrainer(model_type='MVO',
                                            asset_num=asset_num,
                                            input_shape=None,
                                            param_grid=None,
                                            timestamp=timestamp)
        # ew train performance
        rets_m = np.log(price_data).diff()
        rets_m = rets_m.dropna()
        mvo_portfolio.build_benchmark(date_index, ticker_list, rets_m=rets_m, n_train=n_train)
        mvo_metrics, mvo_rets = mvo_portfolio.perform_backtest(prices, initial_amount)

        metrics = pd.concat([metrics, ew_metrics, mvo_metrics], axis=1)
        rets = pd.concat([rets, ew_rets, mvo_rets], axis=1)        

        # save metrics and returns to disk
        metrics.to_csv(f'../data/model_data/{timestamp}/NN_{model_version}_metrics.csv')
        rets.to_csv(f'../data/model_data/{timestamp}/NN_{model_version}_rets.csv')

        # plot and save wealth index
        plot_wealth_index(rets, initial_amount, 'NN_' + model_version, f'../data/model_data/{timestamp}/')


if __name__=="__main__":
   sys.exit(main())