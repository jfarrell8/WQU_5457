# WorldQuant University - MScFE Capstone Project - Group 5457

Repository for the Capstone Project of the Master of science in Financial Engineering program at WorldQuant University.

*Group Members*
- Paulo Mendes
- John Farrell
- Eliab Admasu

### Script Execution

Perform the following actions to execute the portfolio optimization of neural networks and reinforcement learning for dividend aristocrat companies. A few notes first:
1. The config.ini file that is provided allows for adjustment of training and test windows as well as the hyperparameters used in building neural networks and deep learning agents.
2. We have pre-compiled the datasets for the end user. They can be found in the data folder: final_dataset.csv is used for the NN hyperparameter optimization, while rl_dataset.csv is used for the reinforcement learning HPO.
3. We have scripts that detail the process as to how we obtained and transformed the data as needed. Further, there are notebooks that in the sup_notebooks folder that also detail from exploratory data analysis we went through to determine what data was available and how we could incorporate it.

_Project execution instructions_
1. pip install -r /path/to/local/repo/requirements.txt
2. python models_NN.py
3. python models_RL.py

### Data Pre-processing

_data_

This folder contains intermediate and final datasets for our modeling processes. Additionally, it also contains example outputs for NN and RL runs.
- final_dataset.csv is used for the neural network hyperparameter tuning jobs
- rl_dataset.csv is used for the reinforcement learning tuning jobs


_data_wrangling_
- ff_download.py: Script to extract fundamental factor time series data from AlphaVantage
- ff_postprocessing.py: Script to clean up the fundamental factor data from AlphaVantage
- price_data.py: Script to extract price data from YahooFinance for our universe
- yield_curves.py: Script to download daily treasury rates from US Treasury and interpolate yields for analysis

_data_combination.py_
- Builds out the final_dataset.csv and rl_dataset.csv from the data_wrangling outputs. These datasets will be used to perform our modeling.

### Modeling

_models_
- models_NN.py: Performs the hyperparameter tuning for MLP, CNN, and RNN neural nets to identify a portfolio weight distribution that maximizes portfolio Sharpe ratio
- models_RL.py: Performs the hyperparameter tuning for A2C and PPO RL agents using FinRL backbone to also identify a portfolio weight distribution that maximizes portfolio Sharpe ratio

_NN_HPO.ipynb_: Jupyter notebook that provides a hyperparameter tuning setting to capture the optimal weights for the S&P 500 dividend aristocrat portfolio using MLP, CNN, and RNN architectures.

_RL_HPO.ipynb_: Jupyter notebook that provides a hyperparameter tuning setting to capture the optimal weights for the S&P 500 dividen aristocrat portfolio using FinRL (reinforcement learning).

_config.ini_: Provides hyperparameter data and other metadata necessary for executing our data wrangling and modeling runs

_utils_
- utils.py: utility functions use between both models_NN.py and models_RL.py

_sup_notebooks_
- Contains supplementary notebooks used to extract and perform analysis on our datasets
