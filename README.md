# WorldQuant University - MScFE Capstone Project - Group 5457

Repository for the Capstone Project of the Master of science in Financial Engineering program at WorldQuant University.

*Group Members*
- Paulo Mendes
- John Farrell
- Eliab Admasu

Data Pre-processing
_data_
This folder contains intermediate and final datasets for our modeling processes. Additionally, it also contains example outputs for NN and RL runs.
- final_dataset.csv is used for the neural network hyperparameter tuning jobs
- rl_dataset.csv is used for the reinforcement learning tuning jobs


_data_wrangling_
- ff_download.py: Script to extract fundamental factor time series data from AlphaVantage
- ff_postprocessing.py: Script to clean up the fundamental factor data from AlphaVantage
- price_data.py: Script to extract price data from YahooFinance for our universe
- yield_curves.py: Script to download daily treasury rates from US Treasury and interpolate yields for analysis

_data_combination.py_: Builds out the final_dataset.csv and rl_dataset.csv from the data_wrangling outputs. These datasets will be used to perform our modeling.

Modeling
_models_
-models_NN.py: Performs the hyperparameter tuning for MLP, CNN, and RNN neural nets to identify a portfolio weight distribution that maximizes portfolio Sharpe ratio
-models.RL.py: Performs the hyperparameter tuning for A2C and PPO RL agents using FinRL backbone to also identify a portfolio weight distribution that maximizes portfolio Sharpe ratio

_NN_HPO.ipynb_: Jupyter notebook that provides a hyperparameter tuning setting to capture the optimal weights for the S&P 500 dividend aristocrat portfolio using MLP, CNN, and RNN architectures.
_RL_HPO.ipynb_: Jupyter notebook that provides a hyperparameter tuning setting to capture the optimal weights for the S&P 500 dividen aristocrat portfolio using FinRL (reinforcement learning).

_config.ini_: Provides hyperparameter data and other metadata necessary for executing our data wrangling and modeling runs

_utils_
-utils.py: utility functions use between both models_NN.py and models_RL.py

_sup_notebooks_
Contains supplementary notebooks used to extract and perform analysis on our datasets
