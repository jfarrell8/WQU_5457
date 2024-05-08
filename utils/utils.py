import configparser
import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


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


def set_seeds(seed_state):
    np.random.seed(seed_state)
    random.seed(seed_state)
    tf.random.set_seed(seed_state)



def plot_wealth_index(portfolio_returns, initial_amount, timestamp, model_version):
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

    plt.savefig(f'../data/model_data/wealth_index_{model_version}_{timestamp}.png')