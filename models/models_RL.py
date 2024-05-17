import os
import pandas as pd
import numpy as np
from utils.utils import set_seeds, load_config, config_dict_parser, plot_wealth_index
from models_NN import PortfolioModelTrainer
import matplotlib.pyplot as plt
import gym
import datetime
from gym.utils import seeding
from gym import spaces
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3 import A2C, DDPG, PPO, TD3, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.model_selection import ParameterGrid
import sys
import torch


class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                df,
                stock_dim,
                hmax, # max number of shares to trade
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                timestamp,
                agent_name,
                idx,
                turbulence_threshold=None,
                lookback=12,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.agent_name = agent_name
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.timestamp = timestamp
        self.idx = idx

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,))
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list),self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day,:]
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.Date.unique()[0]]


    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['monthly_return']
            plt.plot(df.monthly_return.cumsum(),'r')
            # plt.savefig(f"../data/RL/results/cumulative_reward_{self.timestamp}.png")
            plt.savefig(f"../data/RL/{self.timestamp}/{self.agent_name}/results/cumulative_reward_{self.idx}.png")
            plt.close()

            plt.plot(self.portfolio_return_memory,'r')
            # plt.savefig(f"../data/RL/results/rewards_{self.timestamp}.png")
            plt.savefig(f"../data/RL/{self.timestamp}/{self.agent_name}/results/rewards_{self.idx}.png")
            plt.close()

            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(self.portfolio_value))

            df_monthly_return = pd.DataFrame(self.portfolio_return_memory)
            df_monthly_return.columns = ['monthly_return']
            if df_monthly_return['monthly_return'].std() !=0:
              sharpe = (12**0.5)*df_monthly_return['monthly_return'].mean()/ \
                       df_monthly_return['monthly_return'].std()
              print("Sharpe: ",sharpe)
            print("=================================")

            with open(f"../data/RL/{self.timestamp}/{self.agent_name}/results/performance_{self.idx}.txt", "a") as f:
                f.write("begin_total_asset:{}".format(self.asset_memory[0]))
                f.write("\n")
                f.write("end_total_asset:{}".format(self.portfolio_value))
                f.write("\n")
                f.write(f"Sharpe: {sharpe}")
                f.write("\n")

            return self.state, self.reward, self.terminal,{}

        else:
            #print("Model actions: ",actions)
            # actions are the portfolio weight
            # normalize to sum of 1
            #if (np.array(actions) - np.array(actions).min()).sum() != 0:
            #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
            #else:
            #  norm_actions = actions
            weights = self.softmax_normalization(actions)
            # wts_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # np.savetxt(f"../data/RL/{self.timestamp}/{self.agent_name}/results/wts_{self.idx}_{wts_time}.csv", weights, delimiter=',')

            #print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            
            # calculate portfolio return
            # individual stocks' return * weight
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)

            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # with open(f"../data/RL/{self.timestamp}/{self.agent_name}/results/portfolio_value{self.idx}.txt", "a") as f:
            #     f.write(str(new_portfolio_value))
            #     f.write('\n')

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.Date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value
            #print("Step reward: ", self.reward)
            #self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        # load states
        self.covs = self.data['cov_list'].values[0]
        self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.Date.unique()[0]]

        return self.state

    def render(self, mode='human'):
        return self.state

    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'monthly_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['Date']

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.Ticker.values
        df_actions.index = df_date.Date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

def directory_creator(timestamp, agents):
    for agent in agents:
        if not os.path.exists(f"../data/RL/{timestamp}/{agent}/datasets"):
            os.makedirs(f"../data/RL/{timestamp}/{agent}/datasets")
        if not os.path.exists(f"../data/RL/{timestamp}/{agent}/trained_models"):
            os.makedirs(f"../data/RL/{timestamp}/{agent}/trained_models")
        if not os.path.exists(f"../data/RL/{timestamp}/{agent}/tensorboard_log"):
            os.makedirs(f"../data/RL/{timestamp}/{agent}/tensorboard_log")
        if not os.path.exists(f"../data/RL/{timestamp}/{agent}/results"):
            os.makedirs(f"../data/RL/{timestamp}/{agent}/results")


#Calculate the Sharpe ratio
#This is our objective for tuning
def calculate_sharpe(df):
  if df['monthly_return'].std() !=0:
    sharpe = (12**0.5)*df['monthly_return'].mean()/ \
          df['monthly_return'].std()
    return sharpe
  else:
    return 0


def main():
    # set seeds
    set_seeds(seed_state=42)
    
    # set timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    agents = ['a2c', 'ppo', 'sac']
    
    # create directories if needed
    directory_creator(timestamp, agents)

    # pull in the main dataset
    dataset = pd.read_csv('../data/rl_dataset.csv')
    dataset.Date = pd.to_datetime(dataset.Date)

    # monthly price data
    price_data = pd.read_csv('../data/price_data.csv', index_col='Date')
    price_data.index = pd.to_datetime(price_data.index)
    price_data = price_data.resample('ME').last()
    price_data = price_data.sort_index()
    price_long = price_data.reset_index().melt(id_vars=['Date'], var_name='Ticker', value_name='close')

    # get some params
    config = load_config()
    # trainval_split = float(config.get('NeuralNetParams', 'trainval_split'))
    n_val = int(config.get('NeuralNetParams', 'n_val'))
    initial_amount = int(config.get('MetaData', 'initial_amount'))


    rl_df = dataset
    rl_df.set_index('Date', inplace=True)
    sorted_index = sorted(rl_df.index.unique())
    # let's establish train, validation, and test periods here:
    trainval_date_index = sorted_index[:-n_val]
    test_date_index = sorted_index[-n_val:]
    rl_df.reset_index(inplace=True)
    rl_df.index = rl_df.Date.factorize()[0]

    train_prices = price_data[price_data.index.isin(trainval_date_index)]
    train_prices = train_prices.sort_index()
    test_prices = price_data[price_data.index.isin(test_date_index)]
    test_prices = test_prices.sort_index()


    # add covariance matrix as states
    cov_list = []
    return_list = []
    
    # look back is one year
    lookback=12
    for i in range(lookback, len(rl_df.index.unique())):
        data_lookback = rl_df.loc[i-lookback:i,:]
        return_lookback = data_lookback.pivot_table(index = 'Date',columns = 'Ticker', values = 'ret')
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame({'Date':rl_df.Date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    rl_df = rl_df.merge(df_cov, on='Date')
    rl_df = rl_df.sort_values(['Date','Ticker']).reset_index(drop=True)


    rl_df = price_long.merge(rl_df, on=['Date', 'Ticker']).drop('ret', axis=1)
    rl_df = rl_df.sort_values(by=['Date', 'Ticker'])

    # train_pct = 0.8
    # unique_dates = sorted(rl_df.Date.unique())
    # train_end_date = unique_dates[int(trainval_split * len(unique_dates))]

    rl_train = rl_df.set_index('Date')
    # rl_train = rl_train.loc[:train_end_date, :]
    rl_train = rl_train.loc[:trainval_date_index[-1], :]
    rl_train = rl_train.reset_index()
    rl_train.index = rl_train.Date.factorize()[0]

    rl_test = rl_df.set_index('Date')
    rl_test = rl_test.loc[test_date_index[0]:, :]
    rl_test = rl_test.reset_index()
    rl_test.index = rl_test.Date.factorize()[0]
    # rl_test = rl_test.rename(columns={'Ticker':'tic'})


    stock_dimension = len(rl_train.Ticker.unique())
    state_space = stock_dimension

    agent_best_params = {}

    for agent_name in agents:
        print('#################')
        print('#################')
        print('#################')
        print(agent_name.upper())
        print('#################')
        print('#################')
        print('#################')


        best_params = None
        best_score = -np.inf
        best_idx = None

        param_grid = config_dict_parser(config, f'{agent_name}_param_grid')

        for idx, params in enumerate(ParameterGrid(param_grid)):
            env_kwargs = {
                    "hmax": 100,
                    "initial_amount": initial_amount,
                    "transaction_cost_pct": 0.001,
                    "state_space": state_space,
                    "stock_dim": stock_dimension,
                    "tech_indicator_list": rl_train.columns[3:-2],
                    "action_space": stock_dimension,
                    "reward_scaling": 1e-4,
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "idx": idx
            }

            e_train_gym = StockPortfolioEnv(df = rl_train, **env_kwargs)

            env_train, _ = e_train_gym.get_sb_env()


            agent = DRLAgent(env = env_train)

            model = agent.get_model(model_name=agent_name, model_kwargs = params)

            trained_model = agent.train_model(model=model,
                                            tb_log_name=agent_name,
                                            total_timesteps=5000)

            # trained_model.save(f'../data/RL/{timestamp}/{agent_name}/trained_models/trained_{agent_name}_{idx}.zip')
            trained_model.save(f'../data/RL/{timestamp}/{agent_name}/trained_models/trained_{agent_name}_{idx}.pth')

            # make prediction on the test set to find Sharpe ratio
            e_trade_gym = StockPortfolioEnv(df = rl_test, **env_kwargs)

            df_monthly_return, df_actions = DRLAgent.DRL_prediction(model=trained_model, environment = e_trade_gym)

            df_monthly_return.to_csv(f'../data/RL/{timestamp}/{agent_name}/results/{agent_name}_rets_{idx}.csv')
            df_actions.to_csv(f'../data/RL/{timestamp}/{agent_name}/results/{agent_name}_wts_{idx}.csv')

            sharpe_ratio = calculate_sharpe(df_monthly_return)
            
            if sharpe_ratio > best_score:
                best_score = sharpe_ratio
                best_params = params
                best_idx = idx
        print(agent_name)
        agent_best_params[agent_name] = [best_params, best_score, best_idx]

    train_metrics = pd.DataFrame()
    train_rets = pd.DataFrame()
    test_metrics = pd.DataFrame()
    test_rets = pd.DataFrame()

    for agent_name, values in agent_best_params.items():

        with open(f'../data/RL/{timestamp}/best_params.txt', 'a') as f:
            f.write(str(agent_name) + ': ' + str(values))
            f.write('\n')


        env_kwargs = {
                    "hmax": 100,
                    "initial_amount": initial_amount,
                    "transaction_cost_pct": 0.001,
                    "state_space": state_space,
                    "stock_dim": stock_dimension,
                    "tech_indicator_list": rl_train.columns[3:-2],
                    "action_space": stock_dimension,
                    "reward_scaling": 1e-4,
                    "timestamp": timestamp,
                    "agent_name": agent_name,
                    "idx": idx
            }

        e_train_gym = StockPortfolioEnv(df = rl_train, **env_kwargs)

        env_train, _ = e_train_gym.get_sb_env()


        # agent = DRLAgent(env = env_train)

        # model = agent.get_model(model_name=agent_name, model_kwargs = values[0])

        ####### TEST IN-SAMPLE PERFORMANCE #######
        tuned_model = eval(agent_name.upper()).load(f'../data/RL/{timestamp}/{agent_name}/trained_models/trained_{agent_name}_{values[2]}.pth', env=env_train)
        train_returns, train_wts = DRLAgent.DRL_prediction(model=tuned_model, environment = e_train_gym)


        ####### TEST OUT-OF-SAMPLE PERFORMANCE ########
        test_wts = pd.read_csv(f'../data/RL/{timestamp}/{agent_name}/results/{agent_name}_wts_{values[2]}.csv', index_col='Date')
        test_wts.index = pd.to_datetime(test_wts.index)

        for model_version in ['train', 'test']:
            if model_version == 'train':
                wts = train_wts
                prices = train_prices
            elif model_version == 'test':
                wts = test_wts
                prices = test_prices
            
            portfolio = PortfolioModelTrainer(model_type=agent_name, 
                                            asset_num=None,
                                            input_shape=None,
                                            param_grid=None,
                                            timestamp=timestamp,
                                            wts=wts)
            rl_metrics, rl_rets = portfolio.perform_backtest(prices, initial_amount)

            if model_version == 'train':
                train_metrics = pd.concat([train_metrics, rl_metrics], axis=1)
                train_rets = pd.concat([train_rets, rl_rets], axis=1)
            elif model_version == 'test':
                test_metrics = pd.concat([test_metrics, rl_metrics], axis=1)
                test_rets = pd.concat([test_rets, rl_rets], axis=1)   

    train_rets.to_csv(f'../data/RL/{timestamp}/train_rets.csv')
    test_rets.to_csv(f'../data/RL/{timestamp}/test_rets.csv')
    train_metrics.to_csv(f'../data/RL/{timestamp}/train_metrics.csv')
    test_metrics.to_csv(f'../data/RL/{timestamp}/test_metrics.csv')
    
    plot_wealth_index(train_rets, initial_amount, timestamp, 'train', f'../data/RL/{timestamp}/')
    plot_wealth_index(test_rets, initial_amount, timestamp, 'test', f'../data/RL/{timestamp}/')





if __name__=="__main__":
    sys.exit(main())