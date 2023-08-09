#parameters.py

import os
import pandas as pd
import numpy as np


dataFTSE_MIB = os.path.join('data', 'FTSE_MIB.xlsx')
data = pd.read_excel(dataFTSE_MIB, header=None)
data.sort_index()
mean_price = np.mean(data)
n_stocks = len(data.columns)
rendimenti_set_1 = data.pct_change().dropna()
rendimento_medio_set_1 = rendimenti_set_1.mean()
matrice_covarianza_1 = rendimenti_set_1.cov()
matrice_correlazione_1 = rendimenti_set_1.corr()

R = rendimento_medio_set_1.mean() * 30
ratio_multiple = 0.25
h = 0.06
r_c = 0.07
r_p_multiple = 2
total_cash = 0.05
min_quant = -0.7
max_quant = 0.75
cardin_min = 10
cardin_max = 33
scalar_1 = 1
scalar_2 = 1
scalar_8 = 1
scalar_9 = 1
scalar_12 = 1
scalar_17 = 1
scalar_18 = 1
scalar_quant = 1
scalar_cardinality = 1
min_quant_long = 0.1
max_quant_long = 1
min_quant_short = 0.1
max_quant_short = 1
cardin_min_l_s = 1
cardin_max_l_s = 30
investor_value = 1
T = 0.9
local_min_func_df = pd.DataFrame(index=range(1))
glob_min_func_df = pd.DataFrame(index=range(1))
rule_r_p_df = pd.DataFrame(index=range(1))
rule_r_p_actual_values_df = pd.DataFrame(index=range(1))
rule_R_p_scalar = 1
rule_r_p_actual_values = []
rule_R_p_scalar_values = [rule_R_p_scalar]
quantity_l_s_scalar = 1
quantity_l_s_scalar_values = [quantity_l_s_scalar]
card_l_s_scalar = 1
card_l_s_scalar_values = [card_l_s_scalar]
us_reg_scalar = 1
us_reg_scalar_values = [us_reg_scalar]
l_s_total_values_scalar = 1
l_s_total_values_scalar_values = [l_s_total_values_scalar]
R_P_mean_list = []
R_P_std_list = []
variables_dict = [zip(list(range(4)), [[] for x in range(4)])]