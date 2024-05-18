import configparser
import os
import ast
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta


def load_config():
    config = configparser.ConfigParser()

    # Get the directory path of the current script
    current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent directory
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

    # Construct the path to config.ini in the parent directory
    config_path = os.path.join(parent_directory, 'config.ini')
    config.read(config_path)

    return config

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


def setup_logger(config, filename):
    pass


def set_seeds(seed_state):
    np.random.seed(seed_state)
    random.seed(seed_state)
    tf.random.set_seed(seed_state)



def plot_wealth_index(portfolio_returns, initial_amount, model_version, root_dir):
    # need to insert a row of 0s at the beginning so each line starts at the initial amount
    first_date = portfolio_returns.index[0]
    previous_month_end = first_date - relativedelta(months=1)
    new_row_values = [0] * len(portfolio_returns.columns)
    portfolio_returns = pd.concat([pd.DataFrame([new_row_values], columns=portfolio_returns.columns, index=[previous_month_end]), portfolio_returns], axis=0)

    # now calculate cumulative rets and wealth index
    cumulative_rets = (1 + portfolio_returns).cumprod()
    wealth_index = initial_amount * cumulative_rets

    plt.figure(figsize=(10, 6))
    for column in wealth_index.columns:
        plt.plot(wealth_index.index, wealth_index[column], label=column)

    plt.xlabel('Date')
    plt.ylabel('Wealth Index')
    plt.title('Wealth Index Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(root_dir, f'wealth_index_{model_version}.png'))