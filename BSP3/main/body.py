#body.py

import os 
import argparse
import importlib.machinery
import time
import random
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import scipy.stats as scis
import scipy.optimize as sco
import seaborn as sns
from scipy.stats import norm, zscore
from scipy.stats import entropy
from copy import deepcopy
from deap import algorithms, base, creator, tools

def import_temp(is_temp):
    parameters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters.py")
    if is_temp:
        temp_parameters_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_parameters.py")
        temp_parameters = importlib.machinery.SourceFileLoader("temp_parameters", temp_parameters_path).load_module()
        return temp_parameters
    else:
        parameters = importlib.machinery.SourceFileLoader("parameters", parameters_path).load_module()
        return parameters

# Determine if temp_parameters should be imported
is_temp = os.environ.get("TEMP_PARAMETERS", "False") == "True"

# Import parameters from the appropriate source
parameters = import_temp(is_temp)


#Evolution
def new_objective_function_2_indiv(pop,pareto=True): 
    pop=pop.reshape(1,32)
    fitness=np.zeros((pop.shape[0],1))
    pop_without_cash=deepcopy(pop)
    new_xi=np.zeros((pop_without_cash.shape))
    w_long=np.zeros((pop_without_cash.shape))
    w_short=np.zeros((pop_without_cash.shape))
    df=pd.DataFrame({})
    for solution in range(pop_without_cash.shape[0]):
        local_fitness=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            new_xi[solution,index]=(abs(pop[solution,index]) * mean_price[index])/total_cash
            if pop_without_cash[solution,index]>0:
                w_long[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
            elif pop_without_cash[solution,index]<0:
                w_short[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
    for solution in range(pop_without_cash.shape[0]):
        sum_1=0
        for i in range((pop_without_cash.shape[1]*2)):
            if i>=(pop_without_cash.shape[1]):
                row=i-pop_without_cash.shape[1]
                w_i=w_short[solution,row]
            else:
                row=i
                w_i=w_long[solution,row]
            for j in range((pop_without_cash.shape[1]*2)):  
                if j>=(pop_without_cash.shape[1]):
                    col=j-pop_without_cash.shape[1]
                    w_j=w_short[solution,col]
                else:
                    col=j
                    w_j=w_long[solution,col]
                segma_i_j=matrice_correlazione_1[row][col]
                local_fitness=segma_i_j*w_i*w_j
                sum_1+=local_fitness
        fitness[solution]=sum_1

    global local_min_func_df
    local_min_func_df.insert(len(local_min_func_df.columns), "", fitness.tolist(), True)
    rule_rp,r_p_actual=rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar)
    global rule_r_p_df
    rule_r_p_df.insert(len(rule_r_p_df.columns), "", rule_rp.tolist(), True)
    global rule_r_p_actual_values_df
    rule_r_p_actual_values_df.insert(len(rule_r_p_actual_values_df.columns), "", r_p_actual.tolist(), True)
    df.insert(len(df.columns), "", rule_rp.tolist(), True)
    m=rule_rp.mean()
    global R_P_mean_list
    R_P_mean_list.append(m)
    global R_P_std_list
    R_P_std_list.append(rule_rp.std())
    fitness=fitness+rule_rp
    quantity_l_s=quantity_long_short(pop_without_cash,w_long,w_short,min_quant_long,max_quant_long,min_quant_short,max_quant_short,quantity_l_s_scalar=quantity_l_s_scalar)
    df.insert(len(df.columns), "", quantity_l_s.tolist(), True)
    fitness=fitness+quantity_l_s
    cardinality_l_s=cardinality_long_short(pop_without_cash,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar)
    fitness=fitness+cardinality_l_s
    df.insert(len(df.columns), "", cardinality_l_s.tolist(), True)
    l_s_total_values=Long_short_total_values(pop_without_cash,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar)
    fitness=fitness+l_s_total_values
    df.insert(len(df.columns), "", l_s_total_values.tolist(), True)
    us_reg=usa_regulation(pop_without_cash,w_long,w_short,us_reg_scalar=us_reg_scalar)
    fitness=fitness+us_reg
    df.insert(len(df.columns), "", us_reg.tolist(), True)
    df_ar=df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    fitness_mean = df.mean(axis = 0, skipna = False).to_list()
    if pareto==True:
        pareto_analysis(fitness_mean,95)
    global glob_min_func_df
    glob_min_func_df.insert(len(glob_min_func_df.columns), "", fitness.tolist(), True)
    fit_ser=[]
    for solution in range(pop.shape[0]):
        fit_ser.append(fitness[solution])
    fitness=pd.Series(fit_ser)
    for index,elemnt in enumerate(fitness):
        fitness[index]=elemnt[0]
    fitness=fitness[0]
    return fitness


class Individual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = creator.FitnessMin()
    def __len__(self):
        return len(self.genotype)
    def __getitem__(self, index):
        return self.genotype[index]
    def __setitem__(self, index, value):
        self.genotype[index] = value


def evaluate_population(population):
    fitness_values = []
    for individual in population:
        fitness = new_objective_function_2_indiv(individual.genotype)
        ind_fitness = creator.FitnessMin()
        ind_fitness.values = fitness,
        individual.fitness = ind_fitness
        fitness_values.append(individual.fitness)
    return fitness_values

def main_MYCROSS():
    pop_size=(20, 32)
    population=np.random.uniform(low=-1, high=1, size=pop_size)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Individual, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: np.random.rand(32).reshape((1, 32)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_population)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    n_generations = 20
    variable_evolution = [population.T]
    angle_evolution = []
    entropy_evolution = []
    density_evolution = []
    cost_evolution = []
    population = [Individual(individual) for individual in population]
    for gen in range(n_generations):
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=n_generations)
        fitness_offspring = toolbox.evaluate(offspring)
        for ind, fit in zip(offspring, fitness_offspring):
            ind.fitness.values = fit.values
        population = toolbox.select(population + offspring, k=len(population))
        evaluate_population(population)
        variable_evolution.append(np.array([ind.genotype for ind in population]).T)
        angles = np.arctan2(variable_evolution[gen][1], variable_evolution[gen][0])
        epsilon = 1e-8  
        entropy_vals = entropy(variable_evolution[gen]+ epsilon, axis=1)
        density = np.sum(variable_evolution[-1]) / variable_evolution[-1].size
        angle_evolution.append(angles)
        entropy_evolution.append(entropy_vals)
        density_evolution.append(density)
        total_cost = np.sum([ind.fitness.values[0] for ind in population]) if len(population) > 0 else 0
        cost_evolution.append(total_cost)
    best_individuals = toolbox.select(population, k=5)
    best_variable_values = [ind.genotype for ind in best_individuals]
    variable_evolution = np.array(variable_evolution)
    angle_evolution = np.array(angle_evolution)
    entropy_evolution = np.array(entropy_evolution)
    density_evolution = np.array(density_evolution)
    cost_evolution = np.array(cost_evolution)
    n_variables = variable_evolution.shape[1]
    fig, axes = plt.subplots( 4, 1, figsize=(8, 6))
    axes[0].violinplot(angle_evolution.T, showmedians=True)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Angle')
    axes[1].imshow(entropy_evolution[:n_generations].T, aspect='auto', cmap='hot', origin='lower')
    axes[1].set_ylabel('Entropy')
    axes[2].plot(range(n_generations), density_evolution, marker='o')
    axes[ 2].set_xlabel('Generation')
    axes[2].set_ylabel('Density')
    axes[3].bar(range(n_generations), cost_evolution)
    axes[3].set_xlabel('Generation')
    axes[3].set_ylabel('Cost')        
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.clf()

def plot_vars_auto(x,y1,y2,y3,y4,y5,local_min_func_df,glob_min_func_df,rule_r_p_actual_values_df,rule_r_p_df,num_gen_str,name_of_file='reactive',pareto=True):
    maen_global=glob_min_func_df.values.tolist()[0]
    mean_local=local_min_func_df.values.tolist()[0]
    mean_r_p_panalty_only_lower=rule_r_p_df.values.tolist()[0]
    mean_r_p_actual_values=rule_r_p_actual_values_df.values.tolist()[0]
    global_only_lower=[]
    local_only_lower=[]
    r_p_penalty_only_lower=[]
    r_p_actual_values_only_lower=[]
    smallest=maen_global[0][0]
    for v in maen_global:
        if v[0]<smallest:
            global_only_lower.append(v[0])
            smallest=v[0]
        else:
            global_only_lower.append(smallest)
    smallest=mean_local[0][0]
    for v in mean_local:
        if v[0] <smallest:
            local_only_lower.append(v[0])
            smallest=v[0]
        else:
            local_only_lower.append(smallest)
    smallest=mean_r_p_panalty_only_lower[0][0]
    for v in mean_r_p_panalty_only_lower:
        if v[0]<smallest:
            r_p_penalty_only_lower.append(v[0])
            smallest=v[0]
        else:
            r_p_penalty_only_lower.append(smallest)
    smallest=mean_r_p_actual_values[0][0]
    for v in mean_r_p_actual_values:
        if v[0]<smallest:
            r_p_actual_values_only_lower.append(v[0])
            smallest=v[0]
        else:
            r_p_actual_values_only_lower.append(smallest)
    df_ar=rule_r_p_actual_values_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            if isinstance(df_ar[row,col],list):
                df_ar[row,col]=df_ar[row,col][0]
    rule_r_p_actual_values_df=pd.DataFrame(df_ar)
    df_ar=local_min_func_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            if isinstance(df_ar[row,col],list):
                df_ar[row,col]=df_ar[row,col][0]
    local_min_func_df=pd.DataFrame(df_ar)
    df_ar_g=glob_min_func_df.to_numpy()
    for row in range(df_ar_g.shape[0]):
        for col in range(df_ar_g.shape[1]):
            if isinstance(df_ar_g[row,col],list):
                df_ar_g[row,col]=df_ar_g[row,col][0]
    glob_min_func_df=pd.DataFrame(df_ar_g)
    df_ar=rule_r_p_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            if isinstance(df_ar[row,col],list):
                df_ar[row,col]=df_ar[row,col][0]
    rule_r_p_df=pd.DataFrame(df_ar)
    import os
    if pareto:
        directory = f'/Users/abdulsalamalmohamad/Documents/project 5/codice_finale 2/working/both_work_but_consider_them_one_solution/R_{R}_RATIO_MULTI{ratio_multiple}_h{h}_r_c{r_c}_r_p_multi{r_p_multiple}_cash_{total_cash}'+ 'with_pareto'+"_"+num_gen_str+'/'
    else:
        directory = f'/Users/abdulsalamalmohamad/Documents/project 5/codice_finale 2/working/both_work_but_consider_them_one_solution/R_{R}_RATIO_MULTI{ratio_multiple}_h{h}_r_c{r_c}_r_p_multi{r_p_multiple}'+"_"+num_gen_str+'/'
    os.mkdir(directory)
    if os.path.exists(directory):
        print("Directory created successfully.")
    else:
        print("Failed to create the directory.")
    for i, row in local_min_func_df.iterrows():
        plt.plot(local_min_func_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('LOCAL MIN_FUNC EVOLUTION FOR ALL SOLUTIONS')
    plt.legend()
    plt.savefig(directory+f' pareto___{pareto}____gen_LOCAL_MIN_FUNC_.png', dpi=3000)
    plt.clf()
    for i, row in rule_r_p_actual_values_df.iterrows():
        plt.plot(rule_r_p_actual_values_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('Rule R_P Actual Values')
    plt.legend()
    plt.savefig(directory+f' pareto___{pareto}_Rule_R_p_actual_values_.png', dpi=3000)
    plt.clf()
    x_glob_only_lower=[i for i in range(len(global_only_lower))]
    x_local_only_lower=[i for i in range(len(local_only_lower))]
    x_r_p_penalty_only_lower=[i for i in range(len(r_p_penalty_only_lower))]
    x_r_p_actual_values_only_lower=[i for i in range(len(r_p_actual_values_only_lower))]
    print(len(x_glob_only_lower), len(global_only_lower),x_glob_only_lower, global_only_lower)
    plt.xlabel('ITERATIONS')
    plt.ylabel('GLOBAL MIN_FUNC VALUE ')
    plt.title(' GLOBA LMIN_FUNC EVOLUTION STRICTLY DECREASING')
    plt.plot(x_glob_only_lower, global_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_GLOBAL_MIN_FUNC_EVOLUTION_STRICTLY_DEACREASING.png', dpi=3000)
    plt.clf()
    plt.xlabel('ITERATIONS')
    plt.ylabel('LOCAL MIN_FUNC VALUE ')
    plt.title(' LOCAL_MIN_FUNC EVOLUTION STRICTLY DECREASING')
    plt.plot(x_local_only_lower, local_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_LOCAL_MIN_FUNC_EVOLUTION_STRICTLY_DEACREASING.png', dpi=3000)
    plt.clf()
    plt.xlabel('ITERATIONS')
    plt.ylabel('R_P_ACTUAL_VALUE ')
    plt.title(' R_P ACTUAL EVOLUTION STRICTLY DECREASING')
    plt.plot(x_r_p_actual_values_only_lower, r_p_actual_values_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_R_P_ACTUAL_EVOLUTION_STRICTLY_DEACREASING.png', dpi=3000)
    plt.clf()
    plt.xlabel('ITERATIONS')
    plt.ylabel('R_P_PENALTY_VALUE ')
    plt.title(' R_P PENALTY EVOLUTION STRICTLY DECREASING')
    plt.plot(x_r_p_penalty_only_lower, r_p_penalty_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_R_P_PENALTY_EVOLUTION_STRICTLY_DEACREASING.png', dpi=3000)
    plt.clf()
    for i, row in glob_min_func_df.iterrows():
        plt.plot(glob_min_func_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('GLOBAL MIN_FUNC VALUES ')
        plt.title(' GLOBA LMIN_FUNC EVOLUTION FOR ALL SOLUTIONS')
    plt.legend()
    plt.savefig(directory+f' pareto_{pareto}_GLOBAL_MIN_FUNC.png', dpi=3000)
    plt.clf()
    for i, row in rule_r_p_df.iterrows():
        plt.plot(rule_r_p_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('R_p VALUES ')
        plt.title(' R_p  PENALTY EVOLUTION FOR ALL SOLUTIONS')
    plt.legend()
    plt.savefig(directory+f'_{pareto}_R_p_PENALTY.png', dpi=3000)
    plt.clf()
    x_rp=[i for i in range(len(R_P_mean_list))]
    plt.plot(x_rp, R_P_mean_list, label='MEAN_R_P')
    plt.plot(x_rp, R_P_std_list, label='STANDARD_DEVIATION_R_P')
    plt.legend()
    plt.savefig(directory+f'__{pareto}___MEAN_STD_R_p.png', dpi=3000)
    plt.clf()
    y_glob=glob_min_func_df.mean().to_list()
    x_glob=[i for i in range(len(y_glob))]
    plt.plot(x_glob, y_glob, label='MEAN_GLOB_FUNC')
    y_std=glob_min_func_df.std().to_list()
    plt.plot(x_glob, y_std, label='STD_GLOB_FUNC')
    plt.legend()
    plt.savefig(directory+f'pareto___{pareto}____MEAN_STD_GLOB_FUNC.png', dpi=3000)
    plt.clf()
    y_loc=local_min_func_df.mean().to_list()
    x_loc=[i for i in range(len(y_loc))]
    plt.plot(x_loc, y_loc, label='MEAN_LOC_FUNC')
    y_std=local_min_func_df.std().to_list()
    plt.plot(x_loc, y_std, label='STD_LOC_FUNC')
    plt.legend()
    plt.savefig(directory+f'pareto____{pareto}_____MEAN_STD_LOC_FUN.png', dpi=3000)
    plt.clf()
    plt.plot(x, y1, label='rule_R_p')
    plt.plot(x, y2, label='quantity_l_s')
    plt.plot(x, y3, label='card_l_s')
    plt.plot(x, y4, label='l_s_total_values')
    plt.plot(x, y5, label='us_reg')
    plt.xlabel('ITERATIONS')
    plt.ylabel('VALUES')
    plt.title('PARAMETERS ')
    plt.legend()
    plt.savefig(directory+f'___{pareto}____Parameters.png', dpi=3000)
    plt.clf()


def plot_vars(x,y1,y2,y3,y4,y5,local_min_func_df,glob_min_func_df,rule_r_p_df,rule_r_p_actual_values_df):
    df_ar=rule_r_p_actual_values_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    rule_r_p_actual_values_df=pd.DataFrame(df_ar)
    for i, row in rule_r_p_actual_values_df.iterrows():
        plt.plot(rule_r_p_actual_values_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('Actual_R_P_VALUES EVOLUTION FOR ALL SOLUTIONS')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/RULE_R_P_ACTUAL_VALUES.png', dpi=3000)
    plt.clf()
    df_ar=local_min_func_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    local_min_func_df=pd.DataFrame(df_ar)
    df_ar_g=glob_min_func_df.to_numpy()
    for row in range(df_ar_g.shape[0]):
        for col in range(df_ar_g.shape[1]):
            df_ar_g[row,col]=df_ar_g[row,col][0]
    glob_min_func_df=pd.DataFrame(df_ar_g)
    df_ar=rule_r_p_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    rule_r_p_df=pd.DataFrame(df_ar)
    for i, row in local_min_func_df.iterrows():
        plt.plot(local_min_func_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('LOCAL MIN_FUNC EVOLUTION FOR ALL SOLUTIONS')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/LOCAL_MIN_FUNC.png', dpi=3000)
    plt.clf()
    for i, row in glob_min_func_df.iterrows():
        plt.plot(glob_min_func_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('GLOBAL MIN_FUNC VALUES ')
        plt.title(' GLOBA LMIN_FUNC EVOLUTION FOR ALL SOLUTIONS')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/GLOBAL_MIN_FUNC.png', dpi=3000)
    plt.clf()
    for i, row in rule_r_p_df.iterrows():
        plt.plot(rule_r_p_df.columns[1:], row[1:])
        plt.xlabel('ITERATIONS')
        plt.ylabel('R_p VALUES ')
        plt.title(' R_p EVOLUTION FOR ALL SOLUTIONS')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/R_p.png', dpi=3000)
    plt.clf()
    x_rp=[i for i in range(len(R_P_mean_list))]
    plt.plot(x_rp, R_P_mean_list, label='MEAN_R_P')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/MEAN_R_p.png', dpi=3000)
    plt.clf()
    y_glob=glob_min_func_df.mean().to_list()
    x_glob=[i for i in range(len(y_glob))]
    plt.plot(x_glob, y_glob, label='MEAN_GLOB_FUNC')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/MEAN_GLOB_FUNC.png', dpi=3000)
    plt.clf()
    y_loc=local_min_func_df.mean().to_list()
    x_loc=[i for i in range(len(y_loc))]
    plt.plot(x_loc, y_loc, label='MEAN_GLOB_FUNC')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/MEAN_LOC_FUNC.png', dpi=3000)
    plt.clf()
    plt.plot(x, y1, label='rule_r_p')
    plt.plot(x, y2, label='quantity_l_s')
    plt.plot(x, y3, label='card_l_s')
    plt.plot(x, y4, label='l_s_total_values')
    plt.plot(x, y5, label='us_reg')
    plt.xlabel('ITERATIONS')
    plt.ylabel('VALUES')
    plt.title('PARAMETERS ')
    plt.legend()
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/parameters.png', dpi=3000)
    plt.clf()
    print(f'\n\n\n R is :   {R}')

def get_a_value_df_col(df,col,var):
    for index,row in df['Variables'].iteritems():
        if row==var:
            req_value=df[col][index]
            return req_value
    raise ValueError

def pareto_analysis(values,desired_cumulative_percentage):
    global r_p_multiple
    global ratio_multiple
    data = {
        'Variables': [0,1,2,3,4],
        'Effect': values
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by='Effect', ascending=False)
    total_effect = df['Effect'].sum()
    df['Cumulative Percentage'] = (df['Effect'].cumsum() / total_effect) * 100
    selected_variables = df[df['Cumulative Percentage'] <= desired_cumulative_percentage]
    selected_variables_list=selected_variables['Variables'].to_list()
    unselected_vars=[]

    for index in range(len(df)):
        if df['Variables'][index] not in selected_variables_list:
            unselected_vars.append(df['Variables'][index] )
    total_selected_effect = sum(values)
    selected_variables['Ratio']=['' for x in range(len(selected_variables))]
    selected_variables['Ratio'] = selected_variables['Effect'] / total_selected_effect

    for selected_var in selected_variables_list:
        var_ratio=get_a_value_df_col(selected_variables,'Ratio',selected_var)
        if selected_var==0:
            global rule_R_p_scalar
            rule_R_p_scalar=rule_R_p_scalar*var_ratio*ratio_multiple*r_p_multiple
            r_p_multiple=r_p_multiple*0.5
        elif selected_var==1:
            global quantity_l_s_scalar
            quantity_l_s_scalar=quantity_l_s_scalar*var_ratio*ratio_multiple
        elif selected_var==2:
            global card_l_s_scalar
            card_l_s_scalar=card_l_s_scalar*var_ratio*ratio_multiple
        elif selected_var==3:
            global l_s_total_values_scalar
            l_s_total_values_scalar=l_s_total_values_scalar*var_ratio*ratio_multiple
        elif selected_var==4:
            global us_reg_scalar
            us_reg_scalar=us_reg_scalar*var_ratio*ratio_multiple
    global rule_R_p_scalar_values
    rule_R_p_scalar_values.append(rule_R_p_scalar)
    global quantity_l_s_scalar_values
    quantity_l_s_scalar_values.append(quantity_l_s_scalar)
    global card_l_s_scalar_values
    card_l_s_scalar_values.append(card_l_s_scalar)
    global l_s_total_values_scalar_values
    l_s_total_values_scalar_values.append(l_s_total_values_scalar)
    global us_reg_scalar_values
    us_reg_scalar_values.append(us_reg_scalar)
    new_values=[rule_R_p_scalar,quantity_l_s_scalar,card_l_s_scalar,l_s_total_values_scalar,us_reg_scalar]
    ratio_multiple=ratio_multiple*0.25


def const_eight_nine(pop,R,scalar_8 = scalar_8,scalar_9 = scalar_9):
    solutions_credit=np.zeros((pop.shape[0],1))
    pop_long=np.zeros((pop.shape))
    pop_short=np.zeros((pop.shape))
    for solution in range(pop.shape[0]):
        ret_times_asset=0
        for index,asset in enumerate(pop[solution,:-1]):
            if asset>0:
                pop_long[solution,index]=asset
            elif asset<0:
                pop_short[solution,index]=-asset
    for solution in range(pop.shape[0]):
        sum_long=np.sum(pop_long[solution,:])
        sum_short=np.sum(pop_short[solution,:])
        long_short=sum_long-sum_short
        if long_short!=1:
            local_credit=abs(long_short-1)*scalar_8
            solutions_credit[solution]=local_credit
    nine_credit=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        r_x_long=0
        r_x_short=0
        for index in range(pop.shape[1]-1):
            r_i=rendimento_medio_set_1[index]
            x_l=pop_long[solution,index]
            x_s=pop_short[solution,index]
            r_x_long+=(r_i*x_l)
            r_x_short+=(r_i*x_s)
        r_actual=r_x_long-r_x_short
        if r_actual!=R:
            cred=abs(r_actual-R)*scalar_9
            nine_credit[solution]=cred
    return solutions_credit,nine_credit

def quantity(pop,min_quant,max_quant,scalar_quant=scalar_quant):
    solutions_quantity=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        quant_credit=0
        for index,asset in enumerate(pop[solution,:-1]):
            if asset< min_quant :
                quant_credit+=abs(asset-min_quant)*scalar_quant
            elif asset> max_quant:
                quant_credit+=abs(asset-max_quant)*scalar_quant
        solutions_quantity[solution]=abs(quant_credit)
    return solutions_quantity

def cardinality(pop,cardin_min,cardin_max,scalar_cardinality=scalar_cardinality):
    solutions_cardinality=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        cardinality_local=0
        for asset in pop[solution,:-1]:
            if asset!=0:
                cardinality_local+=1
        if cardinality_local<cardin_min :
            solutions_cardinality[solution,:]=abs(cardinality_local-cardin_min)*scalar_cardinality
        elif  cardinality_local>cardin_max:
            solutions_cardinality[solution,:]=abs(cardinality_local-cardin_max)*scalar_cardinality
    return solutions_cardinality

def rule_one(pop,R,scalar_1=scalar_1):
    fitness_credit=np.zeros((pop.shape[0],1))
    abs_pop=deepcopy(pop)
    for solution in range(pop.shape[0]):
        ret_times_asset=0
        for index in range(pop.shape[1]-1):
            x_i=abs_pop[solution,index]
            r_i=rendimento_medio_set_1[index]
            ret_times_asset+=(x_i*r_i)
        ret_asset_minus_r=ret_times_asset-R
        if ret_asset_minus_r<0:
            fitness_credit[solution]=abs(ret_asset_minus_r)*scalar_1
    return fitness_credit

def rule_2(pop,scalar_2=scalar_2):
    fitness_credit=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        sum_asset=0
        for index in range(pop.shape[1]-1):
            sum_asset+=pop[solution,index]
        sum_asset_minus_1=sum_asset-1
        if sum_asset_minus_1!=0:
            fitness_credit[solution]=abs(sum_asset_minus_1)*scalar_2
    return fitness_credit

def rule_17(pop,scalar_17=scalar_17):
    credit_17=fitness_credit=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        x_cash=pop[solution,pop.shape[1]-1]
        sum_min=0
        for index,asset in enumerate(pop[solution,:-1]):
            sum_min+=np.min(asset,0)
        sum_min=1.03*sum_min
        if x_cash<sum_min:
            credit_17[solution]=abs(x_cash-sum_min)*scalar_17
    return credit_17

def rule_18(pop,scalar_18=scalar_18,r_c=r_c):
    credit_18=fitness_credit=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        sum_assets=0
        for index,asset in enumerate(pop[solution,:-1]):
            if asset>0:
                h_i=0
            else:
                h_i=h
            r_i=rendimento_medio_set_1[index]
            sum_assets+=((r_i-h_i)*r_c)*asset
        if sum_assets<R:
            credit_18[solution]=abs(sum_assets-R)*scalar_18
    return credit_18

def rule_twelve(pop,scalar_12=scalar_12):
    fitness_credit=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        sum_asset=0
        for index in range(pop.shape[1]-1):
            sum_asset+=abs(pop[solution,index])
        sum_asset_minus_2=sum_asset-2
        if sum_asset_minus_2>0:
            fitness_credit[solution]=abs(sum_asset_minus_2) * scalar_12
    return fitness_credit

def new_objective_function(pop,R,min_quant,max_quant,cardin_min,cardin_max):
    pop_without_cash=deepcopy(pop)
    pop_without_cash=pop_without_cash[:,:-1]
    fitness=np.zeros((pop.shape[0],1))
    for solution in range(pop.shape[0]):
        local_fitness=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            for other_index,other_asset in enumerate(pop_without_cash[solution,:]):
                local_fitness+=matrice_correlazione_1[index][other_index]*asset*other_asset
        fitness[solution]=local_fitness
    r_1=rule_one(pop,R,scalar_1)
    fitness=fitness+r_1
    r_2=rule_2(pop,scalar_2)
    fitness=fitness+r_2
    quantity_credit=quantity(pop,min_quant,max_quant,scalar_quant)
    fitness=fitness+quantity_credit
    cardinality_credit=cardinality(pop,cardin_min,cardin_max,scalar_cardinality)
    fitness=fitness+cardinality_credit
    eight,nine=const_eight_nine(pop,R,scalar_8,scalar_9)
    fitness=fitness+eight
    fitness=fitness+nine
    twelve=rule_twelve(pop,scalar_12)
    fitness=fitness+twelve
    r_17=rule_17(pop)
    fitness=fitness+r_17
    r_18= rule_18(pop)
    fitness=fitness+r_18
    fit_ser=[]
    for solution in range(pop.shape[0]):
        fit_ser.append(fitness[solution])
    fitness=pd.Series(fit_ser)
    for index,elemnt in enumerate(fitness):
        fitness[index]=elemnt[0]
    fitness=-fitness
    return fitness

def quantity_long_short(pop_without_cash,w_long,w_short,min_quant_long,max_quant_long,min_quant_short,max_quant_short,quantity_l_s_scalar=quantity_l_s_scalar):
    solutions_quantity=np.zeros((pop_without_cash.shape[0],1))
    for solution in range(pop_without_cash.shape[0]):
        quant_credit=0
        quant_credit_long=0
        quant_credit_short=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            if w_long[solution,index]>0:
                z_i_long=1
            else:
                z_i_long=0
            if w_long[solution,index]>z_i_long*max_quant_long:
                quant_credit_long+=abs(w_long[solution,index]-max_quant_long)
            if w_long[solution,index]<z_i_long*min_quant_long:
                quant_credit_long+=abs(w_long[solution,index]-min_quant_long)
            if w_short[solution,index]>0:
                z_i_short=1
            else:
                z_i_short=0
            if w_short[solution,index]>z_i_short*max_quant_short:
                quant_credit_short+=abs(w_short[solution,index]-max_quant_short)
            elif w_short[solution,index]<z_i_short*min_quant_short:
                quant_credit_short+=abs(w_short[solution,index]-min_quant_short)
            if w_long[solution,index]>z_i_long*max_quant_long:
                quant_credit_long+=abs(w_long[solution,index]-max_quant_long)
            elif w_long[solution,index]<z_i_long*min_quant_long:
                quant_credit_short+=abs(w_long[solution,index]-min_quant_long)
            quant_credit+=(quant_credit_long+quant_credit_short)
        solutions_quantity[solution]=quant_credit*quantity_l_s_scalar
    return solutions_quantity


def cardinality_long_short(pop_without_cash,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar):
    solutions_cardinality=np.zeros((pop_without_cash.shape[0],1))
    for solution in range(pop_without_cash.shape[0]):
        cardinality_local=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            if w_long[solution,index]>0:
                z_i_long=1
            else:
                z_i_long=0
            if w_short[solution,index]>0:
                z_i_short=1
            else:
                z_i_short=0
            cardinality_local+=(z_i_long+z_i_short)
        if cardinality_local<cardin_min_l_s :
            solutions_cardinality[solution,:]=abs(cardinality_local-cardin_min_l_s)* card_l_s_scalar
        elif  cardinality_local>cardin_max_l_s:
            solutions_cardinality[solution,:]=abs(cardinality_local-cardin_max_l_s)* card_l_s_scalar
    return solutions_cardinality


def rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar):
    r_p_penalty=np.zeros((pop_without_cash.shape[0],1))
    r_p_actual=np.zeros((pop_without_cash.shape[0],1))
    for solution in range(pop_without_cash.shape[0]):
        term1=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            r_i=rendimento_medio_set_1[index]
            term1+=(r_i*w_long[solution,index])
            term2=0
            term2+=(r_i*w_short[solution,index])
            term3=0
            if asset>0:
                h_i=0
            else:
                h_i=h
            term3+=(r_c*w_short[solution,index]*h_i)
        r_p_penalty[solution]=(R-(term1-term2+term3)) * rule_R_p_scalar
        r_p_actual[solution]=(term1-term2+term3)
    return r_p_penalty,r_p_actual


def Long_short_total_values(pop_without_cash,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar):
    fitness_l_s=np.zeros((pop_without_cash.shape[0],1))
    for solution in range(pop_without_cash.shape[0]):
        sum_w_l=0
        sum_w_s=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            sum_w_l+=w_long[solution,index]
            sum_w_s+=w_short[solution,index]
        total_sum=sum_w_l-sum_w_s
        fitness_local=abs(total_sum-investor_value)
        if fitness_local>T:
            fitness_l_s[solution]=(abs(fitness_local)-T)*l_s_total_values_scalar
    return fitness_l_s

        
def usa_regulation(pop_without_cash,w_long,w_short,us_reg_scalar=us_reg_scalar):
    fitness_us_reg=np.zeros((pop_without_cash.shape[0],1))
    for solution in range(pop_without_cash.shape[0]):
        sum_w_l=0
        sum_w_s=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            sum_w_l+=w_long[solution,index]
            sum_w_s+=w_short[solution,index]
        total_sum=sum_w_l+sum_w_s
        if total_sum>2:
            fitness_us_reg[solution]=abs(total_sum-2) * us_reg_scalar
    return fitness_us_reg


def new_objective_function_2(pop,R,min_quant,max_quant,cardin_min,cardin_max,pareto=True): 
    fitness=np.zeros((pop.shape[0],1))
    pop_without_cash=deepcopy(pop)
    pop_without_cash=pop_without_cash[:,:-1]
    new_xi=np.zeros((pop_without_cash.shape))
    w_long=np.zeros((pop_without_cash.shape))
    w_short=np.zeros((pop_without_cash.shape))
    df=pd.DataFrame({})
    for solution in range(pop_without_cash.shape[0]):
        local_fitness=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            new_xi[solution,index]=(abs(pop[solution,index]) * mean_price[index])/total_cash
            if pop_without_cash[solution,index]>0:
                w_long[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
            elif pop_without_cash[solution,index]<0:
                w_short[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
    for solution in range(pop_without_cash.shape[0]):
        sum_1=0
        for i in range((pop_without_cash.shape[1]*2)):
            if i>=(pop_without_cash.shape[1]):
                row=i-pop_without_cash.shape[1]
                w_i=w_short[solution,row]
            else:
                row=i
                w_i=w_long[solution,row]
            for j in range((pop_without_cash.shape[1]*2)):  
                if j>=(pop_without_cash.shape[1]):
                    col=j-pop_without_cash.shape[1]
                    w_j=w_short[solution,col]
                else:
                    col=j
                    w_j=w_long[solution,col]
                segma_i_j=matrice_correlazione_1[row][col]
                local_fitness=segma_i_j*w_i*w_j
                sum_1+=local_fitness
        fitness[solution]=sum_1
    if len(fitness)==20:
        global local_min_func_df
        local_min_func_df.insert(len(local_min_func_df.columns), "", fitness.tolist(), True)
    rule_rp,r_p_actual=rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar)
    if len(rule_rp)==20:
        global rule_r_p_df
        rule_r_p_df.insert(len(rule_r_p_df.columns), "", rule_rp.tolist(), True)
        df.insert(len(df.columns), "", rule_rp.tolist(), True)
        m=rule_rp.mean()
        global R_P_mean_list
        R_P_mean_list.append(m)
        global rule_r_p_actual_values_df
        rule_r_p_actual_values_df.insert(len(rule_r_p_actual_values_df), "", r_p_actual.tolist(), True)
    fitness=fitness+rule_rp
    if len(quantity_l_s)==20:
        quantity_l_s=quantity_long_short(pop_without_cash,w_long,w_short,min_quant_long,max_quant_long,min_quant_short,max_quant_short,quantity_l_s_scalar=quantity_l_s_scalar)
        df.insert(len(df), "", quantity_l_s.tolist(), True)
    fitness=fitness+quantity_l_s
    if len(cardinality_l_s)==20:
        cardinality_l_s=cardinality_long_short(pop_without_cash,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar)
        df.insert(len(df), "", cardinality_l_s.tolist(), True)
    fitness=fitness+cardinality_l_s
    l_s_total_values=Long_short_total_values(pop_without_cash,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar)
    if len(cardinality_l_s)==20:
        df.insert(len(df), "", l_s_total_values.tolist(), True)
    fitness=fitness+l_s_total_values
    us_reg=usa_regulation(pop_without_cash,w_long,w_short,us_reg_scalar=us_reg_scalar)
    if len(cardinality_l_s)==20:
        df.insert(len(df), "", us_reg.tolist(), True)
    fitness=fitness+us_reg
    df_ar=df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    fitness_mean = df.mean(axis = 0, skipna = False).to_list()
    if pareto==True:
        pareto_analysis(fitness_mean,95)
    if len(fitness)==20:
        global glob_min_func_df
        glob_min_func_df.insert(len(glob_min_func_df.columns), "", fitness.tolist(), True)
    fit_ser=[]
    for solution in range(pop.shape[0]):
        fit_ser.append(fitness[solution])
    fitness=pd.Series(fit_ser)
    for index,elemnt in enumerate(fitness):
        fitness[index]=elemnt[0]
    fitness=-fitness
    return fitness

def rendimento(pop):
    rendimento_portafoglio=np.dot(rendimento_medio_set_1,pop.T)
    pesi=pop
    rendimento_portafoglio=np.zeros(len(pesi))
    for i in range(len(pesi)):
        rendimento_portafoglio[i] = np.sum(rendimento_medio_set_1 * pesi[i])
    return rendimento_portafoglio


def vol(pop):
    matrice_covarianza_1=rendimenti_set_1.cov()
    pesi=pop
    std_dev_portafoglio=np.zeros(len(pesi)) 
    for i in range(len(pop)):
        std_dev_portafoglio[i]=np.sqrt(np.dot(pesi[i].T,np.dot(matrice_covarianza_1,pesi[i])))
    return std_dev_portafoglio

def risk_parity(pop):
    pesi=pop
    fRP=np.zeros(np.shape(pesi))
    portvar=vol(pesi)**2
    Cx=(np.dot(matrice_covarianza_1,pesi.T))
    for j in range(len(pesi.T)):
            fRP[:,j]=np.abs(((pesi[:,j]*Cx[j,:])/portvar)-(1/geni))
    rp=-np.sum(fRP,axis=1)
    return rp
    
def semivar(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    semivar=(np.var(np.minimum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0),axis=0))
    return semivar

def mean_absolute_deviation(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    mad=np.mean(np.abs(rendimento_portafoglio_mensile-np.mean(rendimento_portafoglio_mensile)))
    return mad

def omega_ratio(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    omega=(np.sum(np.minimum(rendimento_portafoglio_mensile,0),axis=0)/np.sum(np.maximum(rendimento_portafoglio_mensile,0),axis=0))
    return omega

def nuova_rm(pop):
    delta=0.5
    perdita=-0.02
    gain=0.05
    pesi=pop
    conta='meno'
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    if conta=='meno':
        contameno=np.count_nonzero(rendimento_portafoglio_mensile<perdita,axis=0)/len(rendimento_portafoglio_mensile)
        costo=delta*rendimento(pop)-(1-delta)*(contameno)
    elif conta=='piu':
        contapiu=np.count_nonzero(rendimento_portafoglio_mensile>gain,axis=0)/rendimento_portafoglio_mensile
        costo=rendimento(pop)+delta*(contapiu)
    return costo
    
def twosided(pop):
    a=0.25
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    twoside=np.zeros(len(rendimento_portafoglio_mensile.T))
    upside=np.maximum(rendimento_portafoglio_mensile-rendimento_portafoglio_mensile.mean(),0)
    downside=np.maximum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0)
    for z in range(len(rendimento_portafoglio_mensile.T)):
	    twoside[z]=-a*np.linalg.norm(upside.iloc[:,z],ord=1)-(1-a)*np.linalg.norm(downside.iloc[:,z],ord=2)+rendimento_portafoglio_mensile.mean()[z]
    return twoside

def sortino_ratio(pop):
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    semivar=(np.var(np.minimum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0),axis=0))
    sortino=np.mean(rendimento_portafoglio_mensile,axis=0)/semivar
    return sortino

def sharpe(pop):
    return rendimento(pop)/vol(pop)

def mean_variance(pop):
    lambda_1=0.5
    delta=2.0
    return -lambda_1*vol(pop)+(1-lambda_1)*rendimento(pop)-delta*np.sum(abs(pop),axis=1)
    
def mean_semivariance(pop):
    lambda_1=0.5
    return -lambda_1*semivar(pop)+(1-lambda_1)*rendimento(pop)

def mean_mad(pop):
    lambda_1=0.5
    return -lambda_1*mean_absolute_deviation(pop)+(1-lambda_1)*rendimento(pop)

def minimax(pop):
    lambda_1=0.5
    minimax=np.zeros(len(pop))
    for i in range(len(pop)):
        minimax[i]=-lambda_1*min((1-lambda_1)*(rendimento_medio_set_1 * pop[i])-(np.mean(np.abs(rendimenti_set_1-rendimento_medio_set_1))*pop[i]))
    return minimax

def variance_with_skewness(pop):
    lambda_1=0.5
    std_dev_portafoglio=np.zeros(len(pop))
    rendimento_portafoglio=rendimento_medio_set_1@pop.T
    rendimento_portafoglio_mensile=rendimenti_set_1@pop.T
    for i in range(len(pop)):
        std_dev_portafoglio[i]=np.sqrt(np.dot(pop[i].T,np.dot(matrice_covarianza_1,pop[i])))
    dsr2=np.array(np.mean((rendimento_portafoglio_mensile-np.mean(rendimento_portafoglio_mensile))**2))
    dsr3=np.array(np.mean((rendimento_portafoglio_mensile-np.mean(rendimento_portafoglio_mensile))**3))
    ccef=-lambda_1*dsr2+(1-lambda_1)*np.mean(rendimento_portafoglio_mensile)+(dsr3/(dsr2)**(3/2))
    return ccef

def value_at_risk(pop):
    alpha=0.05
    rend=rendimento(pop)
    stdev=vol(pop)
    var=norm.ppf(alpha,rend,stdev)*np.sqrt(21)
    return var

def expected_shortfall(pop):
    alpha=0.05
    rend=rendimento(pop)
    std=vol(pop)
    es=-(alpha**-1*norm.pdf(norm.ppf(alpha))*std - rend)*np.sqrt(21)
    return es

def calcola_entropy(pop):
    aux=np.zeros((cromosomi,geni))
    for i in range(len(pop)):
        for j in range(geni):
            aux[i,j]=-(pop[i,j]/cromosomi)*(np.log(pop[i,j]/cromosomi))/(np.log(2)*geni)
    aux1=np.sum(aux,axis=1)
    aux2=np.sum(aux1,axis=0)
    return aux2

def calcola_pop_fitness(pop):
    obiettivo=new_objective_function_2(pop,R,min_quant,max_quant,cardin_min,cardin_max)
    return obiettivo

def elitist_selection(pop, fitness, num_parents):  
    parents=np.zeros((num_parents, pop.shape[1]))
    for i in range(num_parents):
        posizione_max_fitness = np.where(fitness == np.max(fitness)) 
        posizione_max_fitness = posizione_max_fitness[0][0]
        parents[i,:] = pop[posizione_max_fitness, :]
        fitness[posizione_max_fitness] = -10000
    return parents

def selection(pop, fitness, num_parents):  
    parents=np.zeros((num_parents, pop.shape[1])) 
    parents = pop
    return parents

def tournament_selection(pop,fitness, num_parents):
    parents=np.zeros((num_parents, pop.shape[1])) 
    for i in range(num_parents):
        tournament=np.array(random.choices(range(len(fitness)),k=2))
        fitness_tournament=fitness[tournament]
        stack=np.vstack((tournament,fitness_tournament))
        if stack[1,0]>stack[1,1]:
            posizione=stack[0,0]
        else:
            posizione=stack[0,1]
        parents[i,:]=pop[int(posizione),:]
    return parents

def crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    crossover_point=int(offspring_size[1]/2) 
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0] 
        parent2_pos = (i+1)%parents.shape[0] 
        offspring[i, :crossover_point] = parents[parent1_pos, :crossover_point] 
        offspring[i, crossover_point:] = parents[parent2_pos, crossover_point:] 
    return offspring

def two_point_crossover(parents,offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        crossover_point_1=np.random.randint(1,geni-1)
        crossover_point_2=np.random.randint(crossover_point_1,geni-1)
        if i%2==0:
            offspring[i,:crossover_point_1]=parents[parent1_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent2_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent1_pos,crossover_point_2:] 
        else:
            offspring[i,:crossover_point_1]=parents[parent2_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent1_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent2_pos,crossover_point_2:] 
    return offspring

def three_point_crossover(parents,offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        crossover_point_1=int((offspring_size[1]-1)/3)
        crossover_point_2=int((2*offspring_size[1]-1)/3)
        crossover_point_3=int((3*offspring_size[1]-1)/3)
        if i%2==0:
            offspring[i,:crossover_point_1]=parents[parent1_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent2_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:crossover_point_3]=parents[parent1_pos,crossover_point_2:crossover_point_3]
            offspring[i,crossover_point_3:]=parents[parent2_pos,crossover_point_3:] 
        else:
            offspring[i,:crossover_point_1]=parents[parent2_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent1_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:crossover_point_3]=parents[parent2_pos,crossover_point_2:crossover_point_3]
            offspring[i,crossover_point_3:]=parents[parent1_pos,crossover_point_3:] 
    return offspring

def crossover_uniforme(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0] 
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            r=np.random.uniform(0,1)
            if r>0.5:
                offspring[i,j] = parents[parent1_pos,j] 
            else:
                offspring[i,j] = parents[parent2_pos,j]
    return offspring

def arnone_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0] 
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            r=np.random.uniform(0,1)
            if r>0.5:
                offspring[i,j] =min(parents[parent1_pos,j],parents[parent1_pos,j]*((np.sum(parents[parent1_pos])+np.sum(parents[parent2_pos]))/(2*np.sum(parents[parent1_pos]))))
            else:
                offspring[i,j] =min(parents[parent2_pos,j],parents[parent2_pos,j]*((np.sum(parents[parent1_pos])+np.sum(parents[parent2_pos]))/(2*np.sum(parents[parent2_pos]))))
    return offspring

def crossover_uniforme_globale(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        for j in range(geni):
            offspring[i,j]=random.choice(parents[:,j])
    return offspring

def flat_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =np.random.uniform(min(parents[parent1_pos,j],parents[parent2_pos,j]),max(parents[parent1_pos,j],parents[parent2_pos,j]))
    return offspring

def blend_crossover(parents, offspring_size): 
    alpha=0.5
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                xmin=min(parents[parent1_pos,j],parents[parent2_pos,j])
                xmax=max(parents[parent1_pos,j],parents[parent2_pos,j])
                deltax=xmax-xmin
                offspring[i,j]=np.abs(np.random.uniform(xmin-(alpha*deltax),xmax+(alpha*deltax)))
    return offspring

def average_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =(parents[parent1_pos,j]+parents[parent2_pos,j])/2
    return offspring

def multi_parent_average_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =(parents[parent1_pos,j]+parents[parent2_pos,j]+parents[parent3_pos,j])/3
    return offspring

def gene_pool_crossover(parents,offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0]
        gene_pool=np.hstack((parents[parent1_pos],parents[parent2_pos],parents[parent3_pos]))
        for j in range(geni):
           estrazione=np.random.choice(gene_pool) 
           offspring[i,j]=estrazione
           gene_pool=np.delete(gene_pool,np.where(gene_pool==estrazione)[0][0])
    return offspring

def heuristic_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            w=np.random.uniform(0,1)
            if parents[parent1_pos,j]>parents[parent2_pos,j]:
                offspring[i,j] =parents[parent2_pos,j]+w*(parents[parent1_pos,j]-parents[parent2_pos,j])
            else:
                offspring[i,j] =parents[parent1_pos,j]+w*(parents[parent2_pos,j]-parents[parent1_pos,j])
    return offspring

def arithmetic_crossover(parents, offspring_size): 
    beta=0.7
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j]
            else:
                offspring[i,j]=beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j]
    return offspring

def linear_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        offspring1=np.zeros(geni)
        offspring2=np.zeros(geni)
        offspring3=np.zeros(geni)
        for j in range(geni):
            offspring1[j]=np.abs(0.5*parents[parent1_pos,j]+0.5*parents[parent2_pos,j])
            offspring2[j]=np.abs(1.5*parents[parent1_pos,j]-0.5*parents[parent2_pos,j])
            offspring3[j]=np.abs(-0.5*parents[parent1_pos,j]+1.5*parents[parent2_pos,j])
        offspring_matrix=np.stack((offspring1,offspring2,offspring3))
        offspring_fitness=calcola_pop_fitness(offspring_matrix)
        pos_offspring=np.where(offspring_fitness==max(offspring_fitness))[0][0]
        offspring[i]=offspring_matrix[pos_offspring]
    return offspring

def simulated_binary_crossover(parents,offspring_size): 
    mu=0.05
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0] 
        parent2_pos = (i+1)%parents.shape[0] 
        r=np.random.uniform(0,1)
        if r<0.5:
            beta=(2*r)**(1/(mu+1))
        else:
            beta=(2-2*r)**(-1/(mu+1))
        if i%2==0:
            offspring[i]=np.abs(0.5*((1-beta)*parents[parent1_pos]+(1+beta)*parents[parent2_pos]))
        else:
            offspring[i]=np.abs(0.5*((1+beta)*parents[parent1_pos]+(1-beta)*parents[parent2_pos]))           
    return offspring

def shuffle_crossover(parents,offspring_size): 
    offspring=np.zeros(offspring_size)
    crossover_point=np.random.randint(1,geni-1)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0] 
        parent2_pos = (i+1)%parents.shape[0] 
        parents[parent1_pos]=np.random.permutation(parents[parent1_pos])
        parents[parent2_pos]=np.random.permutation(parents[parent2_pos])
        offspring[i, :crossover_point] = parents[parent1_pos, :crossover_point] 
        offspring[i, crossover_point:] = parents[parent2_pos, crossover_point:] 
    return offspring

def ring_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        aux=np.hstack((parents[parent1_pos],parents[parent2_pos]))
        start=np.random.randint(0,len(aux))
        offspring_aux=[]
        u=np.random.uniform(0,1)
        if u>0.5:
            for j in range(geni):
                offspring_aux.append(aux[(start+j)%aux.shape[0]]) 
        else:
            for j in range(geni):
                offspring_aux.append(aux[(start-j)%aux.shape[0]]) 
        offspring[i]=offspring_aux
    return offspring

def intermediate_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        beta=np.random.uniform(-0.25,1.25)
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=np.abs(beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j])
            else:
                offspring[i,j]=np.abs(beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j])
    return offspring

def line_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    beta=np.random.uniform(-0.25,1.25)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=np.abs(beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j])
            else:
                offspring[i,j]=np.abs(beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j])
    return offspring

def queen_bee_crossover(parents,offspring_size):  
    offspring=np.zeros(offspring_size)
    queen_bee=np.where(calcola_pop_fitness(parents)==max((calcola_pop_fitness(parents))))[0][0]
    parent1_pos=queen_bee
    for i in range(offspring_size[0]):
        parent2_pos=(parent1_pos+(i+1))%parents.shape[0]
        crossover_point_1=np.random.randint(1,geni-1)
        crossover_point_2=np.random.randint(crossover_point_1,geni-1)
        if i%2==0:
            offspring[i,:crossover_point_1]=parents[parent1_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent2_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent1_pos,crossover_point_2:] 
        else:
            offspring[i,:crossover_point_1]=parents[parent2_pos,:crossover_point_1]
            offspring[i,crossover_point_1:crossover_point_2]=parents[parent1_pos,crossover_point_1:crossover_point_2]
            offspring[i,crossover_point_2:]=parents[parent2_pos,crossover_point_2:] 
    return offspring

def laplace_crossover(parents,offspring_size): 
    offspring=np.zeros(offspring_size)
    a=0
    b=5.0
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        alpha=np.random.uniform(0,1)
        if alpha>0.5:
            beta=a-b*np.log(alpha)
        else:
            beta=a+b*np.log(alpha)
        if i%2==0:
            offspring[i]=np.abs(parents[parent1_pos]+beta*np.abs(parents[parent1_pos]-parents[parent2_pos]))
        else:
            offspring[i]=np.abs(parents[parent2_pos]+beta*np.abs(parents[parent1_pos]-parents[parent1_pos]))
    return offspring

def parent_centric_crossover(parents, offspring_size): 
    alpha=0.5
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                a=np.random.uniform(0,min(parents[parent1_pos,j],parents[parent2_pos,j]))
                b=np.random.uniform(max(parents[parent1_pos,j],parents[parent2_pos,j]),1)
                I=np.abs(parents[parent1_pos,j]-parents[parent2_pos,j])
                l=max(a,parents[parent1_pos,j]-I*alpha)
                u=max(b,parents[parent2_pos,j]+I*alpha)
                offspring[i,j]=np.random.uniform(l,u)
    return offspring

def direction_based_crossover(parents, offspring_size):
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        r=np.random.uniform(0,1)
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        vstack=np.vstack((parents[parent1_pos],parents[parent2_pos]))
        if calcola_pop_fitness(vstack)[0]>=calcola_pop_fitness(vstack)[1]:
            offspring[i]=r*(np.abs(parents[parent1_pos]-parents[parent2_pos]))+parents[parent2_pos]
        else:
            offspring[i]=r*(np.abs(parents[parent2_pos]-parents[parent1_pos]))+parents[parent1_pos]
    return offspring

def geometrical_crossover(parents, offspring_size): 
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0] 
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            offspring[i,j] =np.sqrt(parents[parent1_pos,j]*parents[parent2_pos,j])
    return offspring

def sphere_crossover(parents, offspring_size): 
    alpha=0.5
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): 
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            offspring[i,j] =np.sqrt(alpha*parents[parent1_pos,j]+(1-alpha)*parents[parent2_pos,j])
    return offspring

def simplex_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        parent3_pos=(i+2)%parents.shape[0]
        vstack=np.vstack((parents[parent1_pos],parents[parent2_pos],parents[parent3_pos]))
        pos_peggiore_fitness=np.where(calcola_pop_fitness(vstack)==min(calcola_pop_fitness(vstack)))[0][0]
        pos_migliore_fitness=np.where(calcola_pop_fitness(vstack)==max(calcola_pop_fitness(vstack)))[0][0]
        best_parents=np.delete(vstack,(pos_peggiore_fitness),axis=0)
        centroid=np.sum(best_parents,axis=0)/(len(vstack)-1)
        offspring[i]=centroid+(np.abs(centroid-vstack[pos_peggiore_fitness]))
    return offspring

def fuzzy_crossover(parents,offspring_size): 
    offspring=np.zeros(offspring_size)
    d=0.5
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        for j in range(geni):
            if parents[parent1_pos,j]<parents[parent2_pos,j]:
                phi_1=random.triangular(parents[parent1_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j])
                phi_2=random.triangular(parents[parent2_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]) 
            else:
                phi_2=random.triangular(parents[parent1_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent1_pos,j])
                phi_1=random.triangular(parents[parent2_pos,j]-d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]+d*np.abs(parents[parent2_pos,j]-parents[parent1_pos,j]),parents[parent2_pos,j]) 
            offspring[i,j]=random.choice([np.abs(phi_1),np.abs(phi_2)])
    return offspring

def unimodal_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    std1=0.25
    std2=0.05
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0]
        g=0.5*(parents[parent1_pos]+parents[parent2_pos])
        d1=parents[parent1_pos]-g
        d2=parents[parent2_pos]-g
        d3=parents[parent3_pos]-g
        e1=d1/np.abs(d1)
        e2=d2/np.abs(d2)
        e3=d3/np.abs(d3)
        d=parents[parent2_pos]-parents[parent1_pos]
        aux=(np.random.normal(0,std1)*e1*np.abs(d1))+(np.random.normal(0,std1)*e2*np.abs(d2))
        D=np.linalg.norm(parents[parent3_pos]-g)
        offspring[i,:]=np.abs(g+aux+np.random.normal(0,std2)*D*e3)
    return offspring

def parent_centric_normal_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    eta=0.05
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            w=np.random.uniform(0,1)
            if w<0.5:
                offspring[i,j]=np.abs(np.random.normal(parents[parent1_pos,j],np.abs(parents[parent2_pos,j]-parents[parent1_pos,j])/eta))
            else:
                offspring[i,j]=np.abs(np.random.normal(parents[parent2_pos,j],np.abs(parents[parent2_pos,j]-parents[parent1_pos,j])/eta))
    return offspring

def mutation(offspring_crossover,prob_mutation):
    for i in range(offspring_crossover.shape[0]):  
        for j in range(offspring_crossover.shape[1]):
            estraz_outer=random.uniform(0,1)
            if estraz_outer<(prob_mutation):
                estraz_inner=random.uniform(0,1)
                if estraz_inner>0.5:
                    offspring_crossover[i,j]=offspring_crossover[i,j]*1.1
                else:
                    offspring_crossover[i,j]=offspring_crossover[i,j]*0.9
    return offspring_crossover

def gaussian_mutation(offspring_crossover,prob_mutation): 
    for i in range(offspring_crossover.shape[0]):       
        for j in range(offspring_crossover.shape[1]):
            estraz_outer=random.uniform(0,1)
            if estraz_outer<(prob_mutation):
                offspring_crossover[i,j]=max(min(1,np.random.normal(offspring_crossover[i,j],1)),0)               
    return offspring_crossover

def uniform_mutation(offspring_crossover,prob_mutation):
    for i in range(offspring_crossover.shape[0]):
        for j in range(offspring_crossover.shape[1]):
            estraz_outer=random.uniform(0,1)
            if estraz_outer<(prob_mutation):
                offspring_crossover[i,j]=np.random.uniform(0,1)         
    return offspring_crossover

def lista_crossover(parents,offspring_size):
    c1=crossover(parents,offspring_size)
    c2=crossover_uniforme(parents,offspring_size)
    c3=crossover_uniforme_globale(parents,offspring_size)
    c4=flat_crossover(parents,offspring_size)
    c5=blend_crossover(parents,offspring_size)
    c6=average_crossover(parents,offspring_size)
    c7=simulated_binary_crossover(parents,offspring_size)
    c9=intermediate_crossover(parents,offspring_size)
    c10=geometrical_crossover(parents,offspring_size)
    c11=arithmetic_crossover(parents,offspring_size)
    c12=laplace_crossover(parents,offspring_size)
    c13=two_point_crossover(parents,offspring_size)
    c14=queen_bee_crossover(parents,offspring_size)
    c16=ring_crossover(parents,offspring_size)
    c17=parent_centric_crossover(parents,offspring_size)
    c18=heuristic_crossover(parents,offspring_size)
    c19=three_point_crossover(parents,offspring_size)
    c20=line_crossover(parents,offspring_size)
    c21=sphere_crossover(parents,offspring_size)
    c22=multi_parent_average_crossover(parents,offspring_size)
    c23=gene_pool_crossover(parents,offspring_size)
    c24=linear_crossover(parents,offspring_size)
    c25=simplex_crossover(parents,offspring_size)
    c26=arnone_crossover(parents,offspring_size)
    return c1,c2,c3,c4,c5,c6,c7,c10,c11,c12,c13,c14,c18,c19,c20


def lista_crossover2(parents,offspring_size):
    aux_list=[crossover(parents,offspring_size),crossover_uniforme(parents,offspring_size),heuristic_crossover(parents,offspring_size),
              laplace_crossover(parents,offspring_size),queen_bee_crossover(parents,offspring_size),two_point_crossover(parents,offspring_size),
                arithmetic_crossover(parents,offspring_size),geometrical_crossover(parents,offspring_size),
                simulated_binary_crossover(parents,offspring_size),average_crossover(parents,offspring_size),
                blend_crossover(parents,offspring_size),flat_crossover(parents,offspring_size),
                crossover_uniforme_globale(parents,offspring_size),three_point_crossover(parents,offspring_size),
                linear_crossover(parents,offspring_size),direction_based_crossover(parents,offspring_size),unimodal_crossover(parents,offspring_size),
              fuzzy_crossover(parents,offspring_size),simplex_crossover(parents,offspring_size),parent_centric_normal_crossover(parents,offspring_size)]
    return aux_list


listone=[crossover,crossover_uniforme,heuristic_crossover,
              laplace_crossover,two_point_crossover,
                arithmetic_crossover,geometrical_crossover,
                simulated_binary_crossover,average_crossover,
                blend_crossover,flat_crossover,
                crossover_uniforme_globale,three_point_crossover,
                linear_crossover,direction_based_crossover,unimodal_crossover,
              fuzzy_crossover,parent_centric_normal_crossover]

geni=len
cromosomi=20
num_parents=5 
simulazioni=1
num_generazioni=1000
pop_size=(cromosomi,geni)
contatore_fitness_max=[]
contatore_fitness_media=[]
contatore_entropy=[]
start=time.time()
window=10
num_crossover=18
fitness_modulo1=np.zeros((num_generazioni,num_crossover))
entropy_modulo1=np.zeros((num_generazioni,num_crossover))
deltafitness_modulo1=np.zeros((num_generazioni-window,num_crossover))
deltaentropy_modulo1=np.zeros((num_generazioni-window,num_crossover))
credit_reward_list=np.zeros((num_generazioni-window,num_crossover))
credit_reward_aggregation=np.zeros((num_generazioni-2*window,num_crossover))
ap_selection=np.zeros((num_generazioni-2*window,num_crossover))
ap_selection[0,:]=np.ones(num_crossover)/num_crossover

def aggregated_criteria_computation(fitness_modulo1,entropy_modulo1):
    if i>=window:
        delta_fitness_media=pd.DataFrame(fitness_modulo1[i-window:i,:]).diff()
        delta_entropy=pd.DataFrame(entropy_modulo1[i-window:i,:]).diff()
        Fwin1=delta_fitness_media.mean()
        Fwin2=delta_entropy.mean()
        deltafitness_modulo1[i-window,:]=Fwin1
        deltaentropy_modulo1[i-window,:]=Fwin2
        return Fwin1, Fwin2, deltafitness_modulo1

def reward_computation(theta):
    store_reward=np.zeros(num_crossover)
    if i>=window:
        for s in range(len(store_reward)):
            origine=[0,0]
            Fwin1=aggregated_criteria_computation(fitness_modulo1,entropy_modulo1)[0][s]
            Fwin2=aggregated_criteria_computation(fitness_modulo1,entropy_modulo1)[1][s] 
            x=np.linspace(0,np.max(1.5*Fwin1),100)
            m=np.tan(theta) #slope
            y=m*x
            dp=abs((Fwin2-(m*Fwin1)))/(np.sqrt(1+m**2)) 
            dpp=np.sqrt((Fwin1-origine[0])**2+(Fwin2-origine[1])**2) 
            reward=np.sqrt(dpp**2-dp**2)
            store_reward[s]=reward
        return store_reward
prova=np.zeros(1000)

def increasing_strategy():
    angle=0
    if i>=window:
        if i<=num_generazioni/4:
            angle=0
        elif i>=num_generazioni/4 and i<=num_generazioni/2:
            angle=np.pi/6
        elif i>=num_generazioni/2 and i<=num_generazioni*(3/4):
            angle=np.pi/3
        elif i>=num_generazioni*(3/4):
            angle=np.pi/2
    return angle

def decreasing_strategy():
    angle=np.pi/2
    if i>=window:
        if i<=num_generazioni/4:
            angle=np.pi/2
        elif i>=num_generazioni/4 and i<=num_generazioni/2:
            angle=np.pi/3
        elif i>=num_generazioni/2 and i<=num_generazioni*(3/4):
            angle=np.pi/6
        elif i>=num_generazioni*(3/4):
            angle=0
    return angle

def always_moving_strategy():
    angle=np.pi/2
    if i>=window:
        if i<=num_generazioni/5:
            angle=np.pi/2
        elif i>=num_generazioni*(1/5) and i<=num_generazioni*(2/5):
            angle=0
        elif i>=num_generazioni*(2/5) and i<=num_generazioni*(3/5):
            angle=np.pi/2
        elif i>=num_generazioni*(3/5) and i<=num_generazioni*(4/5):
            angle=0
        elif i>=num_generazioni*(4/5):
            angle=np.pi/2
    return angle

def reactive_moving_strategy():
    angle=0
    if i>=window:
        if  ((entropy_list[i-1]-entropy_list[i-window])/(entropy_list[i-window]))<(-1/100):
            angle=0
        elif np.abs((fitness_media[i-1]-fitness_media[i-window])/(fitness_media[i-window]))<(1/100):
            angle=np.pi/2
        else:
            angle=0
    return angle

def credit_assignment(strategy):
    if i>=window:
        angle=strategy
        store_reward=reward_computation(angle)
        if len(store_reward)>=18:
            prova[i]=store_reward[17]
        else:
            prova[i]=store_reward[len(store_reward)-1]
        credit_reward_list[i-window,:]=store_reward
    if i>=2*window:
        credit_reward_aggregation[i-2*window,:]=np.mean(pd.DataFrame(credit_reward_list[i-2*window:i-window,:]).dropna())
    return credit_reward_aggregation

def operator_selection(credit_function):
    choose_op='pm'
    if choose_op=='pm':
        p_min=0.01
        K=num_crossover
        idx=np.random.randint(0,19)
        credito=credit_function
        if i>=2*window:
            wheel_selection=p_min+(1-K*p_min)*(credito[i-2*window,:]/(np.sum(credito[i-2*window,:])))
            wheel_selection=np.cumsum(wheel_selection)
            u=np.random.uniform(0,1)
            for c in range(len(wheel_selection)):
                if u<wheel_selection[c]:
                    idx=c
                    break
    elif choose_op=='mab':
        idx=i
        credito=credit_function
        C=0.00001
        aux=np.sum(matrice_memoria_operatori,axis=1)
        if i>=2*window:
            print(credito[i-2*window,:])
            mab_selection=credito[i-2*window,:]+C*np.sqrt(np.log(np.sum(aux))/aux)
            c=np.where(mab_selection==max(mab_selection))
            idx=c[0][0]
            for j in range(num_crossover):
                ph.add_element(credito[i-2*window,j])
                if ph.detected_change():
                    print('restart')
    elif choose_op=='ap':
        beta=0.5
        idx=np.random.randint(0,19)
        p_min=0.01
        p_max=1-(num_crossover-1)*p_min
        credito=credit_function
        if i>=2*window and i<num_generazioni:
            best_ip=np.where(credito[i-2*window,:]==max(credito[i-2*window,:]))
            best_ip=best_ip[0][0]
            best_ip_succ=np.where(credito[i-2*window+1,:]==max(credito[i-2*window+1,:]))
            best_ip_succ=best_ip_succ[0][0]
            for j in range(num_crossover):
                ap_selection[i-2*window+1,j]=ap_selection[i-2*window,j]+beta*(p_min-ap_selection[i-2*window,j])
            ap_selection[i-2*window+1,best_ip_succ]=ap_selection[i-2*window,best_ip]+beta*(p_max-ap_selection[i-2*window,best_ip])
    return idx
      

def gen_portifolio(cromosomi,geni):
    solutions=np.zeros((cromosomi,geni))
    for solution in range(solutions.shape[0]):
        for gene in range(geni):
            units_short=random.random.uniform(0,1)
    pass