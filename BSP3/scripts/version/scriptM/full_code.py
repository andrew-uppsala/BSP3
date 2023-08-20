import numpy as np
import pandas as pd
import random
#import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import random
import seaborn as sns
import scipy.optimize as sco
import time
import scipy.stats as scis

#from skmultiflow.drift_detection import PageHinkley
from scipy.stats import norm, zscore
#from operators import *
from copy import deepcopy
# import numpy as np
# import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from scipy.stats import entropy

# from cross_risk import *
#sortino ratio
#differnt objective functions
#just risk fitness penalty 
#risk for my strategy 
#risk for scalars 
""" 
>>>>>>variable settings lines: 56-124 
>>>>>>>rules functions 82-254
>>>>>>> the outer call of the objective function (wher we chose which one we want and i added mine : search for [def calcola_pop_fitness(pop)]  )
>>>>>>> the actual  objective function (the one we made,within this one you we enforce the rules by the calling there corresponding function): search for [ def new_objective_function(pop,R,min_quant,max_quant,cardin_min,cardin_max) ]  )
>>>>>>>> 
"""
#set_stock_prices_1=['T','V','MCD','INTC','AAPL','AMZN','KO','PFE','GOOG','AFL','EBAY','MO','JNJ','WFC','PEP','KHC','ABBV','AMD','BAC','GS']
#set_stock_prices_1=['KO','GOOG']
#data=web.DataReader(set_stock_prices_1,data_source="yahoo",start='08/01/2015',end='08/01/2020')['Adj Close']
#data=pd.read_csv('C:\\Users\\nonloso\\Desktop\\codici_ga\\FTSE_MIB1.CSV',header=None)
# data=pd.read_excel('/Users/abdulsalamalmohamad/Documents/project 5/codice_finale 2/FTSE_MIB.xlsx',header=None)
data=pd.read_excel('/Users/abnerandreymartinezzamudio/Downloads/BSP3/BSP3/data/FTSE_MIB.xlsx',header=None)
data.index = pd.to_datetime(data.index)
# data.index = pd.DatetimeIndex(data.index)
# data.index = pd.to_datetime(data.index)
# #print(data.columns)
# raise ValueError
# rendimenti_set_1 = (data.pct_change() + 1).fillna(0) - 1
# rendimenti_set_1 = rendimenti_set_1.dropna()
# rendimenti_set_1.index = range(len(rendimenti_set_1))
# #print(pd.__version__)
# #print(np.__version__)
# raise ValueError
data.reset_index(drop=True, inplace=True)
mean_price = np.mean(data, axis=0)
#data=data.resample('M').apply(lambda x: x[-1])
# data.reset_index(drop=True, inplace=True)
# data.sort_index(inplace=True)
mean_price = np.mean(data, axis=0)
# #print(mean_price[5])
# raise ValueError

n_stocks=len(data.columns)

rendimenti_set_1=data.pct_change().dropna()
# ##print(rendimenti_set_1)
# raise ValueError
rendimento_medio_set_1=rendimenti_set_1.mean() 
# #print(rendimento_medio_set_1)
# raise ValueError
matrice_covarianza_1=rendimenti_set_1.cov()

matrice_correlazione_1=rendimenti_set_1.corr()
# ##print(matrice_correlazione_1)
# raise ValueError


#begin first Paper
""" here begins my part where each function is a constraint named accordingly
scalar 12 coresponds to equation 12 in the paper and so on  """

#expected return 
""" paper 1 parameters """
# Portfolio Selection via Particle Swam Optimization: state-of-the-art 
#name code : all scalars are weights of constraints_named accordingly for example scalar_1 is the scalar of equation 1
# R=rendimento_medio_set_1.mean()/10
# R_set=False
#Parameter R
R=7.998607E-05






# R_asset=0
# #prin
# t(R)
# raise ValueError
ratio_multiple=0.7
h=0.06
r_c=0.07
r_p_multiple=0.7
# #print(R)
# raise ValueError
total_cash=0.05
min_quant=-0.7
max_quant=0.75

cardin_min=10
cardin_max=33
scalar_1=1
scalar_2=1
scalar_8=1
scalar_9=1
scalar_12=1
scalar_17=1
scalar_18=1
scalar_quant=1
scalar_cardinality=1


""" paper 2 needed extra parameters """
# Genetic algorithm with short selling .....Parameters values



# Genetic algorithm with short selling .....2.3
min_quant_long=0.1
max_quant_long=1

min_quant_short=0.1
max_quant_short=1

# Genetic algorithm with short selling .....2.2
cardin_min_l_s=1
cardin_max_l_s=30

# Genetic algorithm with short selling .....2.6
investor_value=1
# Genetic algorithm with short selling .....2.6
#trialsa and compare
T=0.8

# df = pd.DataFrame(columns=['Column1', 'Column2', 'Column3'])
df_constraints=pd.DataFrame(columns=['Column1', 'Column2', 'Column3'])
local_min_func_df=pd.DataFrame(index=range(1))
glob_min_func_df=pd.DataFrame(index=range(1))
rule_r_p_df=pd.DataFrame(index=range(1))
rule_r_p_actual_values_df=pd.DataFrame(index=range(1))
n_col=30
df_r_p_porti = pd.DataFrame(columns=[f'Asset_{i+1}' for i in range(n_col+2)])
# df_r_p_porti.columns[-4:] = ['Return','Penalty' ,'Required_R','Global_Fitness']
new_column_names = {
    'Asset_29': 'Return',
    'Asset_30': 'Penalty',
    'Asset_31': 'Required_R',
    'Asset_32': 'Global_Fitness (After Constrains)'
}
df_r_p_porti = df_r_p_porti.rename(columns=new_column_names)


# new_column_names = {'Asset_{}'.format(i): name for i, name in enumerate(['Fitness', 'Required_R', 'Global_Fitness'], start=-2)}

# # Rename the last three columns
# df_r_p_porti = df_r_p_porti.rename(columns=new_column_names)





# df_r_p_porti.columns[-3:] = ['Fitness', 'Required_R','Global_Fitness']
df_risk_porti = pd.DataFrame(columns=[f'Asset_{i+1}' for i in range(n_col+1)])
# df_risk_porti.columns[-3:] = ['Finess' ,'Required_R','Global_Fitness_of_R_Individual_Right_After']

new_column_names = {
    'Asset_29': 'Finess',
    'Asset_30': 'Required_R',
    'Asset_31': 'Global_Fitness_of_R_Individual_Right_After',
    
}
df_risk_porti = df_risk_porti.rename(columns=new_column_names)



# new_column_names = {'Asset_{}'.format(i): name for i, name in enumerate(['Fitness', 'Required_R', 'Global_Fitness'], start=-2)}

# Rename the last three columns
# df_r_p_porti = df_r_p_porti.rename(columns=new_column_names)
# df_r_p_porti.columns[-4:] = ['Return', 'Required_R','Penalty','Global_Fitness']
# local_min_func_df_only_lower=pd.DataFrame(index=range(1))
# glob_min_func_df=pd.DataFrame(index=range(1))
# # rule_r_p_df=pd.DataFrame(index=range(1))
# rule_r_p_actual_values_df=pd.DataFrame(index=range(1))
# Genetic algorithm with short selling .....scalar of 2.1.2

# global rule_R_p_scalar
rule_R_p_scalar=1

# global rule_R_p_scalar_values
# rule_R_p_scalar_values=[rule_R_p_scalar]
# rule_R_p_values=[rule_R_p_scalar]
rule_r_p_actual_values=[]
rule_R_p_scalar_values=[rule_R_p_scalar]
# temp=[rule_R_p_scalar]

# Genetic algorithm with short selling .....scalar of 2.3
# global quantity_l_s_scalar
quantity_l_s_scalar=1

# global quantity_l_s_scalar_values
quantity_l_s_scalar_values=[quantity_l_s_scalar]

# Genetic algorithm with short selling .....scalar of 2.2
# global card_l_s_scalar
card_l_s_scalar=1

# global card_l_s_scalar_values
card_l_s_scalar_values=[card_l_s_scalar]

# Genetic algorithm with short selling .....scalar of 2.7
# global us_reg_scalar
us_reg_scalar=1

# Genetic algorithm with short selling .....scalar of 2.6





# global us_reg_scalar_values
us_reg_scalar_values=[us_reg_scalar]


# global l_s_total_values_scalar
l_s_total_values_scalar=1

# global l_s_total_values_scalar_values
l_s_total_values_scalar_values=[l_s_total_values_scalar]

R_P_mean_list=[]
R_P_std_list=[]

# global variables
# variables=[rule_R_p_scalar,quantity_l_s_scalar,card_l_s_scalar,l_s_total_values_scalar,us_reg_scalar]
# (rule_R_p_scalar_values,quantity_l_s_scalar_values,card_l_s_scalar_values,l_s_total_values_scalar_values,us_reg_scalar_values)

# global variables_dict
# variables_dict=[zip(list(range(4)), variables)]

variables_dict=[zip(list(range(4)), [[] for x in range(4)])]
'''
R_p cossing below




'''

# Define the CustomIndividual class
# class CustomIndividual:
#     def __init__(self):
#         self.genotype = None
#         self.fitness = None
# class CustomIndividual:
#     def __init__(self, genotype, fitness):
#         self.genotype = genotype
#         self.fitness = fitness

# Define the rule_R_p function
def rule_R_p(pop, pop_without_cash, new_xi, w_long, w_short, rule_R_p_scalar=rule_R_p_scalar):
    # #print(pop_without_cash.shape)
    # pop_without_cash=pop_without_cash.reshape(1,28)
    global R
    r_p_penalty = np.zeros((pop_without_cash.shape[0], 1))
    r_p_actual = np.zeros((pop_without_cash.shape[0], 1))
    #print(pop_without_cash)
    # raise  ValueError
    for solution in range(pop_without_cash.shape[0]):
        #print('solution number',solution)
        term1 = 0
        term2 = 0
        term3 = 0
        for index, asset in enumerate(pop_without_cash[solution, :]):

            r_i = rendimento_medio_set_1[index]

            term1 += (r_i * w_long[solution, index])
            term2 += (r_i * w_short[solution, index])
            if term2 > 1 or r_i>1 or w_long[solution, index]>1 or w_short[solution, index]>1:
                
                #print(f'w_short {w_short[solution, index]},r_i {r_i},term2 {term2}')
                #print(f'w_long {w_long[solution, index]},r_i {r_i},term1 {term1}')
                #print(f'w_long{w_long}')
                #print(f'w_short{w_short}')
                raise ValueError

            if asset > 0:
                h_i = 0
            else:
                h_i = h
            
            term3 += (r_c * w_short[solution, index] * h_i)

            


            
            
            if R > 1:
                print('R is greater than 2',R)
                raise ValueError
        # getcontext().prec = 30
        
        # result = (Decimal(str(R)) - (Decimal(str(term1)) - Decimal(str(term2)) + Decimal(str(term3)))) * Decimal(str(rule_R_p_scalar))
        # result = (Decimal(str(R)) - (Decimal(str(term1)) - Decimal(str(term2)) + Decimal(str(term3)))) 
        result=R-(term1-term2+term3)
        if math.isinf(result):
            r_p_penalty[solution] = float('inf')
        else:
            r_p_penalty[solution] = result

        r_p_actual[solution] = term1 - term2 + term3
        #print(f'r_p_penalty {r_p_penalty} r_p_actual {r_p_actual}')
        # raise ValueError
    return r_p_penalty, r_p_actual


# Rest of your code









from deap import base, creator, tools, algorithms
import numpy as np

from deap.base import Fitness





class CustomFitness(base.Fitness):
    weights = (-1, 5.0)  # Set the weights for minimization and maximization

    def __init__(self):
        super().__init__()

    def __hash__(self):
        return hash(str(tuple(self.wvalues)))




class CustomIndividual:
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = CustomFitness()

    def __hash__(self):
        return hash(tuple(self.genotype.flatten()))

    def __eq__(self, other):
        return np.array_equal(self.genotype, other.genotype)

    def __len__(self):
        return len(self.genotype)

def evaluate_individual_r_p(individual):
    genotype = individual.genotype
    fitness_penalty, fitness_actual = rule_R_p(genotype, genotype, new_xi, w_long, w_short, rule_R_p_scalar)
    return fitness_penalty, fitness_actual

# def mutGaussian(individual, mu, sigma, indpb, negative_prob):
#     for i in range(len(individual.genotype)):
#         if random.random() < indpb:
#             if random.random() < negative_prob:
#                 individual.genotype[i] += random.gauss(-mu, sigma)  # Allow negative values
#                 individual.genotype[i] = np.minimum(np.maximum(individual.genotype[i], -1.0), 0.0)  # Limit negative values to -1 to 0
#             else:
#                 individual.genotype[i] += random.gauss(mu, sigma)  # Allow positive values
#                 individual.genotype[i] = np.minimum(np.maximum(individual.genotype[i], 0.0), 1.0)  # Limit positive values to 0 to 1
#     return individual,

def mutGaussian(individual, mu, sigma, indpb, negative_prob):
    for i in range(len(individual.genotype)):
        if random.random() < indpb:
            if random.random() < negative_prob:
                individual.genotype[i] += random.gauss(-mu, sigma)  # Allow negative values
                individual.genotype[i] = np.clip(individual.genotype[i], -1.0, 1.0)  # Limit values to -1 to 1
            else:
                individual.genotype[i] += random.gauss(mu, sigma)  # Allow positive values
                individual.genotype[i] = np.clip(individual.genotype[i], -1.0, 1.0)  # Limit values to -1 to 1
    return individual,






def main_MYCROSS_r_p(population, n_generations=10):
    creator.create("Individual", CustomIndividual)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.Individual(np.random.rand(1, 28)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=20)
    toolbox.register("evaluate", evaluate_individual_r_p)
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", mutGaussian, mu=0.0, sigma=0.2, indpb=0.1)
    toolbox.register("mutate", mutGaussian, mu=0.5, sigma=0.4, indpb=0.2, negative_prob=0.1)
    toolbox.register("select", tools.selNSGA2)

    population = toolbox.population(n=1)

    for gen in range(n_generations):
        #print(f"\n\n\n\nGENERATION >>>>>>>>>> {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fitness_offspring = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitness_offspring):
            ind.fitness.values = (fit[0], fit[1])  # Assign penalty and actual fitness values

        population = toolbox.select(population + offspring, k=len(population))

    best_individual = tools.selBest(population, k=1)[0]
    #print(f"Best Individual: {best_individual}")
    #print(f"Best Individual Fitness Values: {best_individual.fitness.values}")
    # #print(f"Best Individual Actual Fitness: {best_individual.fitness.actual}")
    best_genotype = np.array(best_individual.genotype)
    if best_individual.fitness.values:
        best_fitness = best_individual.fitness.values[0]
    else:
        best_fitness = None
    # rule_rp = best_individual.fitness.actual
    penalty, actual = rule_R_p(best_genotype, best_genotype, new_xi, w_long, w_short, rule_R_p_scalar)

    return best_genotype, best_fitness, penalty, actual







# pop = np.random.rand(1, 28)
# n_generations = 34  # Number of generations

# # Define other necessary variables (pop_without_cash, new_xi, w_long, w_short, rule_R_p_scalar)

# best_genotype, best_fitness, rule_rp = main_MYCROSS_r_p(pop, n_generations)






# penalty, actual = rule_R_p(best_genotype, best_genotype, new_xi, w_long, w_short, rule_R_p_scalar)







'''
risk crossing below

'''

def risk_function(pop):
   
    pop_without_cash=deepcopy(pop)

    for solution in range(pop_without_cash.shape[0]):
        sum_1=0
        for i in range((pop_without_cash.shape[1]*2)):
            if i>=(pop_without_cash.shape[1]):
                row=i-pop_without_cash.shape[1]
                w_i=w_short[solution,row]
            else:
                row=i
                w_i=w_long[solution,row]
            # w_i=new_xi[solution,row]

            for j in range((pop_without_cash.shape[1]*2)):  
                if j>=(pop_without_cash.shape[1]):
                    col=j-pop_without_cash.shape[1]
                    w_j=w_short[solution,col]
                else:
                    col=j
                    w_j=w_long[solution,col]

                # w_j=new_xi[solution,col]
                segma_i_j=matrice_correlazione_1[row][col]
                local_fitness=segma_i_j*w_i*w_j
                # ##print(w_i,w_j,segma_i_j,local_fitness)
                sum_1+=local_fitness
        # ##print(sum_1)
        #returns the variance of the portfolio without taking into account the constraint violations

        fitness[solution]=sum_1
        fitness[solution] = tuple(fitness[solution])
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

def mutate(individual, mu, sigma, indpb, negative_prob):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            if random.random() < negative_prob:
                individual[i] += random.gauss(-mu, sigma)  # Allow negative values
                individual[i] = np.minimum(np.maximum(individual[i], -1.0), 0.0)  # Limit negative values to -1 to 0
            else:
                individual[i] += random.gauss(mu, sigma)  # Allow positive values
                individual[i] = np.minimum(np.maximum(individual[i], 0.0), 1.0)  # Limit positive values to 0 to 1
    return individual,







def evaluate_population_risk(population):
    fitness_values = []
    for individual in population:
        # genotype = individual.genotype.reshape((1, 28))
        fitness = risk_function(individual.genotype)
        ind_fitness = creator.FitnessMin()
        ind_fitness.values = tuple(fitness.flatten())  # Convert fitness to a hashable tuple
        individual.fitness = ind_fitness
        fitness_values.append(individual.fitness)
    return fitness_values


def main_MYCROSS_risk(external_individual):
    pop_size = (1, 28)  # Size of the population
    n_generations = 3  # Number of generations

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Individual, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: np.random.rand(28).reshape((1, 28)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual,n=20)
    toolbox.register("evaluate", evaluate_population_risk)
    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", mutate, mu=0.0, sigma=0.2, indpb=0.1)
    toolbox.register("mutate", mutate, mu=0.5, sigma=0.2, indpb=0.1, negative_prob=0.2)

    # toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.1, negative_prob=0.2)
    toolbox.register("select", tools.selNSGA2)

    # Create the initial population with the external individual
    population = toolbox.population(n=pop_size[0])
    population[0].genotype = external_individual

    for gen in range(n_generations):
        #print(f"\n\n\n\nGENERATION >>>>>>>>>> {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        # Evaluate the objective function for the offspring
        fitness_offspring = toolbox.evaluate(offspring)

        # Assign the fitness values to each offspring individual
        for ind, fit in zip(offspring, fitness_offspring):
            ind.fitness.values = fit.values

        # Merge the population and offspring
        population = toolbox.select(population + offspring, k=len(population))

        # Call the evaluate_population function on the updated population
        evaluate_population_risk(population)

        # Calculate the angle, entropy, and density of the solutions
        # angles = np.arctan2(population[0].genotype[1], population[0].genotype[0])
        # epsilon = 1e-8  # Small constant to avoid zero probabilities

        # Calculate the fitness for the best individual
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]

    return best_individual.genotype, best_fitness











"""MY EVOlUTION Strategy"""
# def risk_function(pop):
#     pop=pop.reshape(1,28)
#     # ##print(pop.shape)
#     # raise ValueError()
#     #Genetic algorithm with short selling .....2.1
#     fitness=np.zeros((pop.shape[0],1))


#     pop_without_cash=deepcopy(pop)
#     # pop_without_cash=pop_without_cash.T
#     # pop_without_cash=pop_without_cash[:,:-1]

#     new_xi=np.zeros((pop_without_cash.shape))
#     # weights=deepcopy(pop)
#     w_long=np.zeros((pop_without_cash.shape))
#     w_short=np.zeros((pop_without_cash.shape))
    
#     # #print(pop_without_cash.shape)
#     # raise ValueError
#     df=pd.DataFrame({})

#     for solution in range(pop_without_cash.shape[0]):
#         local_fitness=0
#         for index,asset in enumerate(pop_without_cash[solution,:]):
           
            
#             #print( len(mean_price))
#             # raise ValueError
#             #print(index)
#             new_xi[solution,index]=(abs(pop[solution,index]) * mean_price[index])/total_cash
#             # ##print(new_xi[solution,index])
#             if pop_without_cash[solution,index]>0:
#                 w_long[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
#             elif pop_without_cash[solution,index]<0:
#                 w_short[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash

#     # ##print(w_long[0])
#     # raise ValueError

    
#     for solution in range(pop_without_cash.shape[0]):
#         sum_1=0
#         for i in range((pop_without_cash.shape[1]*2)):
#             if i>=(pop_without_cash.shape[1]):
#                 row=i-pop_without_cash.shape[1]
#                 w_i=w_short[solution,row]
#             else:
#                 row=i
#                 w_i=w_long[solution,row]
#             # w_i=new_xi[solution,row]

#             for j in range((pop_without_cash.shape[1]*2)):  
#                 if j>=(pop_without_cash.shape[1]):
#                     col=j-pop_without_cash.shape[1]
#                     w_j=w_short[solution,col]
#                 else:
#                     col=j
#                     w_j=w_long[solution,col]

#                 # w_j=new_xi[solution,col]
#                 segma_i_j=matrice_correlazione_1[row][col]
#                 local_fitness=segma_i_j*w_i*w_j
#                 # ##print(w_i,w_j,segma_i_j,local_fitness)
#                 sum_1+=local_fitness
#         # ##print(sum_1)
#         #returns the variance of the portfolio without taking into account the constraint violations

#         fitness[solution]=sum_1
#     return fitness




def new_objective_function_2_indiv(pop,pareto=False): 
    global list_const
    list_const=[]
    pop=pop.reshape(1,28)
    # ##print(pop.shape)
    # raise ValueError()
    #Genetic algorithm with short selling .....2.1
    global fitness
    fitness=np.zeros((pop.shape[0],1))


    pop_without_cash=deepcopy(pop)
    # pop_without_cash=pop_without_cash.T
    # pop_without_cash=pop_without_cash[:,:-1]
    global new_xi
    new_xi=np.zeros((pop_without_cash.shape))
    # weights=deepcopy(pop)
    global w_long
    w_long=np.zeros((pop_without_cash.shape))
    global w_short
    w_short=np.zeros((pop_without_cash.shape))
    
    # #print(pop_without_cash.shape)
    # raise ValueError
    global df
    global R
    df=pd.DataFrame({})

    for solution in range(pop_without_cash.shape[0]):
        local_fitness=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
           
            
            #print( len(mean_price))
            # raise ValueError
            # #print(index)
            new_xi[solution,index]=(abs(pop_without_cash[solution,index]) * rendimento_medio_set_1[index])/total_cash
            # new_xi[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash

            # ##print(new_xi[solution,index])
            if pop_without_cash[solution,index]>0:
                w_long[solution,index]=(abs(pop_without_cash[solution,index]) * rendimento_medio_set_1[index])/total_cash
            elif pop_without_cash[solution,index]<0:
                w_short[solution,index]=(abs(pop_without_cash[solution,index]) * rendimento_medio_set_1[index])/total_cash

    pop,fitness=main_MYCROSS_risk(pop)
    print(f'risk penalty of all solutions is : \n {fitness}')
    #print('fitness',fitness)
    # raise ValueError
    pop = pop.reshape((1, 28))
    row_porti = pop.flatten().tolist()
    row_porti.append(fitness)
    row_porti.append(R)
    # global df_risk_porti
    # df_risk_porti.loc[len(df_risk_porti)] = row_porti
    
    # #print(type(pop),pop.shape,fitness)
    # raise ValueError
    global local_min_func_df
    local_min_func_df.insert(len(local_min_func_df.columns), "", fitness.tolist(), True)
    # ##print(local_min_func_df.columns)
    ##print('>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n',local_min_func_df,'>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n')
        # raise ValueError
    ##print("min_function Paper 2",fitness)
    # Genetic algorithm with short selling .....2.1 R_p
    best_genotype,best_fitness, rule_rp, r_p_actual=main_MYCROSS_r_p(pop)
    best_genotype = best_genotype.reshape((1, 28))
    row_r_p = best_genotype.flatten().tolist()
    row_r_p.append(r_p_actual[0][0])
    row_r_p.append(rule_rp[0][0])
    if R>1:
        #print(f'R {R}')
        raise ValueError
    row_r_p.append(R)
    # ['Return','Penalty' ,'Required_R','Global_Fitness']
    # global df_r_p_porti
    # df_r_p_porti.loc[len(df_r_p_porti)] = row_r_p
    # #print(row_r_p,rule_rp[0][0],r_p_actual[0][0])
    # raise ValueError
    # raise ValueError
    # rule_rp,r_p_actual=rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar)
    # df.insert(2, "Age", [21, 23, 24, 21], True)
    # ##print('rule_r_p is ',rule_rp)
    # if len(rule_rp)==20:
    global rule_r_p_df
    # df_ar_g=rule_rp.to_numpy()
    # for row in range(df_ar_g.shape[0]):
    #     for col in range(df_ar_g.shape[1]):
    #         if isinstance(df_ar_g[row,col],list):
    #             df_ar_g[row,col]=df_ar_g[row,col][0]
    # rule_nums=pd.DataFrame(df_ar_g)
    rule_r_p_df.insert(len(rule_r_p_df.columns), "", rule_rp.tolist(), True)
    global rule_r_p_actual_values_df
    rule_r_p_actual_values_df.insert(len(rule_r_p_actual_values_df.columns), "", r_p_actual.tolist(), True)
    df.insert(len(df.columns), "", rule_rp.tolist(), True)

    # global rule_r_p_actual_values
    # rule_r_p_actual_values.append(r_p_actual)



    m=rule_rp.mean()
    global R_P_mean_list
    R_P_mean_list.append(m)
    global R_P_std_list
    R_P_std_list.append(rule_rp.std())
    ##print(f"\n\n\n\n R_P_MEAN is >>>>>>>>>>>>>>>>>{m} \n\n\n\n ")
    fitness=fitness+rule_rp
    # ##print('fitness is : ',fitness)
    # raise ValueError
    # neg_rp=-rule_rp
    # ##print(rule_rp.shape)
    # df.insert(len(df), "", rule_rp.tolist(), True)
    

    # Genetic algorithm with short selling......2.3
    quantity_l_s=quantity_long_short(best_genotype,w_long,w_short,min_quant_long,max_quant_long,min_quant_short,max_quant_short,quantity_l_s_scalar=quantity_l_s_scalar)
    ##print('quantity_long_short Paper 2 is ',quantity_l_s)
    ##print(type(quantity_l_s),quantity_l_s.shape,len(df))
    df.insert(len(df.columns), "", quantity_l_s.tolist(), True)
    # ##print(quantity_l_s)
    # ##print(df)
    # raise ValueError

    # raise ValueError
    fitness=fitness+quantity_l_s


    # Genetic algorithm with short selling......2.2
    cardinality_l_s=cardinality_long_short(best_genotype,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar)
    ##print('cardinality_l_s Paper 2 is ',cardinality_l_s)
    fitness=fitness+cardinality_l_s
    df.insert(len(df.columns), "", cardinality_l_s.tolist(), True)


    # Genetic algorithm with short selling......2.6
    l_s_total_values=Long_short_total_values(best_genotype,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar)
    ##print('l_s_total_values Paper 2 is ',l_s_total_values)
    fitness=fitness+l_s_total_values
    df.insert(len(df.columns), "", l_s_total_values.tolist(), True)
    
    # Genetic algorithm with short selling......2.7 
    us_reg=usa_regulation(best_genotype,w_long,w_short,us_reg_scalar=us_reg_scalar)
    ##print('us_reg Paper 2 is ',us_reg)
    fitness=fitness+us_reg
    df.insert(len(df.columns), "", us_reg.tolist(), True)
    # print(f'df is {df}')
    # global df_constraints
    # df_constraints = pd.concat([df_constraints, df], ignore_index=True)
    # print(df_constraints)
    # print(df)
    # raise ValueError
    df_ar=df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    # df_to_add = pd.DataFrame(df_ar)
    # cons_scalar_list=[rule_R_p_scalar,quantity_l_s_scalar,]
    global df_constraints
    df_constraints.loc[len(df_constraints)] = list_const
    # df_constraints = pd.concat([df_constraints, df_to_add], ignore_index=True)
    # print('list const',list_const)
    # print('df_const',df_constraints.head)
    # raise ValueError
    ##print(df)
    ##print(len(df.columns))
    ##print(df_ar.shape)
    # ##print(df_ar[0,:])
    # df=df[1:]
    # df.columns=[0,1,2,3,4]
    # ##print(df.columns )
    # df.drop(0, inplace=True, axis=0)
    ##print('>>>>>>>>',df)
    #print(df.values.tolist())
    list_of_lists=df.values.tolist()
    df_list=[]
    for x in list_of_lists[0]:
        if not isinstance(x,float):
            df_list.append(x[0])
        else:
            df_list.append(x)
        #df_list.append(x[0])
    # df_list=[x[0] for x in list_of_lists]s
    # #print(type(df))
    # #print(df_list)
    # raise ValueError
    # fitness_mean = df.mean(axis = 0, skipna = False).to_list()
    fitness_mean = np.average(df_list)
    
    if pareto==True:
        
        pareto_analysis(df_list,97)
    
    global glob_min_func_df
    glob_min_func_df.insert(len(glob_min_func_df.columns), "", fitness.tolist(), True)
    # ##print()
    fit_ser=[]
    for solution in range(pop.shape[0]):
        fit_ser.append(fitness[solution])
    
    fitness=pd.Series(fit_ser)

    for index,elemnt in enumerate(fitness):
        # ##print('element is ',elemnt)
        fitness[index]=elemnt[0]
        # raise ValueError
    
    # Genetic algorithm with short selling......final fitness after all constraints
    
    # fitness=-fitness
    # ##print(fitness)
    fitness=fitness[0]
    print(f'fitness of all solutions is : \n {fitness}')
    row_r_p.append(fitness)
    global df_r_p_porti
    df_r_p_porti.loc[len(df_r_p_porti)] = row_r_p


    row_porti.append(fitness)
    global df_risk_porti
    df_risk_porti.loc[len(df_risk_porti)] = row_porti
    # ##print(fitness)
    # raise ValueError 
    return fitness












# def new_objective_function_2_indiv(pop,pareto=False): 
#     # ##print(pop.shape)
#     # pop=pop.reshape(1,32)
#     pop=pop.reshape(1,28)
#     # ##print(pop.shape)
#     # raise ValueError()
#     #Genetic algorithm with short selling .....2.1
#     fitness=np.zeros((pop.shape[0],1))


#     pop_without_cash=deepcopy(pop)
#     # pop_without_cash=pop_without_cash.T
#     # pop_without_cash=pop_without_cash[:,:-1]

#     new_xi=np.zeros((pop_without_cash.shape))
#     # weights=deepcopy(pop)
#     w_long=np.zeros((pop_without_cash.shape))
#     w_short=np.zeros((pop_without_cash.shape))
    
#     # #print(pop_without_cash.shape)
#     # raise ValueError
#     df=pd.DataFrame({})

#     for solution in range(pop_without_cash.shape[0]):
#         local_fitness=0
#         for index,asset in enumerate(pop_without_cash[solution,:]):
           
            
#             #print( len(mean_price))
#             # raise ValueError
#             #print(index)
#             new_xi[solution,index]=(abs(pop[solution,index]) * mean_price[index])/total_cash
#             # ##print(new_xi[solution,index])
#             if pop_without_cash[solution,index]>0:
#                 w_long[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
#             elif pop_without_cash[solution,index]<0:
#                 w_short[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash

#     # ##print(w_long[0])
#     # raise ValueError

    
#     for solution in range(pop_without_cash.shape[0]):
#         sum_1=0
#         for i in range((pop_without_cash.shape[1]*2)):
#             if i>=(pop_without_cash.shape[1]):
#                 row=i-pop_without_cash.shape[1]
#                 w_i=w_short[solution,row]
#             else:
#                 row=i
#                 w_i=w_long[solution,row]
#             # w_i=new_xi[solution,row]

#             for j in range((pop_without_cash.shape[1]*2)):  
#                 if j>=(pop_without_cash.shape[1]):
#                     col=j-pop_without_cash.shape[1]
#                     w_j=w_short[solution,col]
#                 else:
#                     col=j
#                     w_j=w_long[solution,col]

#                 # w_j=new_xi[solution,col]
#                 segma_i_j=matrice_correlazione_1[row][col]
#                 local_fitness=segma_i_j*w_i*w_j
#                 # ##print(w_i,w_j,segma_i_j,local_fitness)
#                 sum_1+=local_fitness
#         # ##print(sum_1)
#         #returns the variance of the portfolio without taking into account the constraint violations

#         fitness[solution]=sum_1
#     # Genetic algorithm with short selling......2.1 
#     # if len(fitness)==20:

#     global local_min_func_df
#     local_min_func_df.insert(len(local_min_func_df.columns), "", fitness.tolist(), True)
#     # ##print(local_min_func_df.columns)
#     ##print('>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n',local_min_func_df,'>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n')
#         # raise ValueError
#     ##print("min_function Paper 2",fitness)
#     # Genetic algorithm with short selling .....2.1 R_p

#     rule_rp,r_p_actual=rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar)
#     # df.insert(2, "Age", [21, 23, 24, 21], True)
#     # ##print('rule_r_p is ',rule_rp)
#     # if len(rule_rp)==20:
#     global rule_r_p_df
#     # df_ar_g=rule_rp.to_numpy()
#     # for row in range(df_ar_g.shape[0]):
#     #     for col in range(df_ar_g.shape[1]):
#     #         if isinstance(df_ar_g[row,col],list):
#     #             df_ar_g[row,col]=df_ar_g[row,col][0]
#     # rule_nums=pd.DataFrame(df_ar_g)
#     rule_r_p_df.insert(len(rule_r_p_df.columns), "", rule_rp.tolist(), True)
#     global rule_r_p_actual_values_df
#     rule_r_p_actual_values_df.insert(len(rule_r_p_actual_values_df.columns), "", r_p_actual.tolist(), True)
#     df.insert(len(df.columns), "", rule_rp.tolist(), True)

#     # global rule_r_p_actual_values
#     # rule_r_p_actual_values.append(r_p_actual)



#     m=rule_rp.mean()
#     global R_P_mean_list
#     R_P_mean_list.append(m)
#     global R_P_std_list
#     R_P_std_list.append(rule_rp.std())
#     ##print(f"\n\n\n\n R_P_MEAN is >>>>>>>>>>>>>>>>>{m} \n\n\n\n ")
#     fitness=fitness-rule_rp
#     # ##print('fitness is : ',fitness)
#     # raise ValueError
#     # neg_rp=-rule_rp
#     # ##print(rule_rp.shape)
#     # df.insert(len(df), "", rule_rp.tolist(), True)
    

#     # Genetic algorithm with short selling......2.3
#     quantity_l_s=quantity_long_short(pop_without_cash,w_long,w_short,min_quant_long,max_quant_long,min_quant_short,max_quant_short,quantity_l_s_scalar=quantity_l_s_scalar)
#     ##print('quantity_long_short Paper 2 is ',quantity_l_s)
#     ##print(type(quantity_l_s),quantity_l_s.shape,len(df))
#     df.insert(len(df.columns), "", quantity_l_s.tolist(), True)
#     # ##print(quantity_l_s)
#     # ##print(df)
#     # raise ValueError

#     # raise ValueError
#     fitness=fitness+quantity_l_s


#     # Genetic algorithm with short selling......2.2
#     cardinality_l_s=cardinality_long_short(pop_without_cash,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar)
#     ##print('cardinality_l_s Paper 2 is ',cardinality_l_s)
#     fitness=fitness+cardinality_l_s
#     df.insert(len(df.columns), "", cardinality_l_s.tolist(), True)


#     # Genetic algorithm with short selling......2.6
#     l_s_total_values=Long_short_total_values(pop_without_cash,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar)
#     ##print('l_s_total_values Paper 2 is ',l_s_total_values)
#     fitness=fitness+l_s_total_values
#     df.insert(len(df.columns), "", l_s_total_values.tolist(), True)
    
#     # Genetic algorithm with short selling......2.7 
#     us_reg=usa_regulation(pop_without_cash,w_long,w_short,us_reg_scalar=us_reg_scalar)
#     ##print('us_reg Paper 2 is ',us_reg)
#     fitness=fitness+us_reg
#     df.insert(len(df.columns), "", us_reg.tolist(), True)

#     df_ar=df.to_numpy()
#     for row in range(df_ar.shape[0]):
#         for col in range(df_ar.shape[1]):
#             df_ar[row,col]=df_ar[row,col][0]
#     ##print(df)
#     ##print(len(df.columns))
#     ##print(df_ar.shape)
#     # ##print(df_ar[0,:])
#     # df=df[1:]
#     # df.columns=[0,1,2,3,4]
#     # ##print(df.columns )
#     # df.drop(0, inplace=True, axis=0)
#     ##print('>>>>>>>>',df)
#     #print(df.values.tolist())
#     list_of_lists=df.values.tolist()
#     df_list=[]
#     for x in list_of_lists[0]:
#         df_list.append(x[0])
#     # df_list=[x[0] for x in list_of_lists]s
#     # #print(type(df))
#     # #print(df_list)
#     # raise ValueError
#     # fitness_mean = df.mean(axis = 0, skipna = False).to_list()
#     fitness_mean = np.average(df_list)
#     # #print('>>>>>>>> real mean',fitness_mean)
#     # raise ValueError


#     ##print('>>>>>>>> real mean',fitness_mean)
#     # fitness_mean[0]=0
    
#     ##print('>>>>>>>> -R_p mean',fitness_mean)
#     # ##print(len(fitness))
#     # raise ValueError
#     if pareto==True:
#         # pareto_analysis(fitness_mean,97)
#         pareto_analysis(df_list,97)
#     # raise ValueError


#     # raise ValueError
#     #overall cost function of paper :Genetic algorithm with short selling ...


#     # ##print(len(fitness))
#     # raise ValueError
#     # if len(fitness)==20:
#     global glob_min_func_df
#     glob_min_func_df.insert(len(glob_min_func_df.columns), "", fitness.tolist(), True)
#     # ##print()
#     fit_ser=[]
#     for solution in range(pop.shape[0]):
#         # ##print(fitness[solution])
#         fit_ser.append(fitness[solution])
#     # fit_ser=np.array(fit_ser)
#     # ##print("final fitness",fitness)
#     fitness=pd.Series(fit_ser)

#     for index,elemnt in enumerate(fitness):
#         # ##print('element is ',elemnt)
#         fitness[index]=elemnt[0]
#         # raise ValueError
    
#     # Genetic algorithm with short selling......final fitness after all constraints
    
#     # fitness=-fitness
#     # ##print(fitness)
#     fitness=fitness[0]
#     # ##print(fitness)
#     # raise ValueError 
#     return fitness
# def objective_function(individual):
#     """Your objective function implementation."""
#     # Assuming individual is a 1D array representing an individual solution
#     # Perform evaluation based on the equation and return a single scalar value
#     return np.sum(individual)

# class Individual:
#     def __init__(self, genotype):
#         self.genotype = genotype
#         self.fitness = creator.FitnessMin()

#     def __len__(self):
#         return len(self.genotype)

#     def __getitem__(self, index):
#         return self.genotype[index]

#     def __setitem__(self, index, value):
#         self.genotype[index] = value


# def evaluate_population(population):
#     fitness_values = []
#     for individual in population:
#         fitness = new_objective_function_2_indiv(individual.genotype)
#         ind_fitness = creator.FitnessMin()
#         ind_fitness.values = fitness,
#         individual.fitness = ind_fitness
#         fitness_values.append(individual.fitness)
#     return fitness_values
# def main_MYCROSS():
#     pop_size=(20, 32)
#     # population = np.random.rand(20, 32)
#     population=np.random.uniform(low=-1, high=1, size=pop_size)
#     # Create the optimization problem
#     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#     creator.create("Individual", Individual, fitness=creator.FitnessMin)

#     toolbox = base.Toolbox()
#     toolbox.register("individual", tools.initIterate, creator.Individual, lambda: np.random.rand(32).reshape((1, 32)))
#     # toolbox.register("individual", tools.initIterate, creator.Individual, np.random.rand, size=32)
#     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#     toolbox.register("evaluate", evaluate_population)
#     toolbox.register("mate", tools.cxTwoPoint)
#     toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.1)
#     toolbox.register("select", tools.selNSGA2)

#     n_generations = 20

#     # Initialize a list to store the variable values, angle, entropy, and density over generations
#     variable_evolution = [population.T]
#     angle_evolution = []
#     entropy_evolution = []
#     density_evolution = []
#     cost_evolution = []
#     # Convert the initial population to the custom Individual class
#     population = [Individual(individual) for individual in population]

#     for gen in range(n_generations):
#         # offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
#         # offspring = algorithms.pso(population, toolbox, ngen=1)
#         # offspring = algorithms.eaDE(population, toolbox, F=0.5, CR=0.3, ngen=1)
#         algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=n_generations)

#         # Example usage of MOEA/D
#         # algorithms.eaMOEA(population, toolbox, ngen=n_generations)
#         # Evaluate the objective function for the offspring
#         fitness_offspring = toolbox.evaluate(offspring)

#         # Assign the fitness values to each offspring individual
#         for ind, fit in zip(offspring, fitness_offspring):
#             ind.fitness.values = fit.values

#         # Merge the population and offspring
#         population = toolbox.select(population + offspring, k=len(population))

#         # Call the evaluate_population function on the updated population
#         evaluate_population(population)

#         # Append the variable values to the evolution list
#         variable_evolution.append(np.array([ind.genotype for ind in population]).T)

#         # Calculate the angle, entropy, and density of the solutions
#         angles = np.arctan2(variable_evolution[gen][1], variable_evolution[gen][0])
         
#         epsilon = 1e-8  # Small constant to avoid zero probabilities
#         entropy_vals = entropy(variable_evolution[gen]+ epsilon, axis=1)
       
# # entropy_vals = entropy(variable_evolution[-1] + epsilon, axis=1)

#         # angles = np.arctan2(variable_evolution[-1][1], variable_evolution[-1][0])
#         # entropy_vals = entropy(variable_evolution[-1])
#         density = np.sum(variable_evolution[-1]) / variable_evolution[-1].size

#         # Append the angle, entropy, and density values to the respective evolution lists
#         angle_evolution.append(angles)
#         entropy_evolution.append(entropy_vals)
#         density_evolution.append(density)

#         # Calculate the total cost of the population
#         total_cost = np.sum([ind.fitness.values[0] for ind in population]) if len(population) > 0 else 0
#         cost_evolution.append(total_cost)


#     best_individuals = toolbox.select(population, k=5)
#     best_variable_values = [ind.genotype for ind in best_individuals]

#     ##print(len(best_variable_values))
    

#     # Convert the variable_evolution, angle_evolution, entropy_evolution, and density_evolution lists to NumPy arrays
#     variable_evolution = np.array(variable_evolution)
#     angle_evolution = np.array(angle_evolution)
#     entropy_evolution = np.array(entropy_evolution)
#     density_evolution = np.array(density_evolution)
#     cost_evolution = np.array(cost_evolution)

#     ##print(entropy_evolution)
#     # raise ValueError()
#     # Plot the evolution of variable values, angle, entropy, and density
#     n_variables = variable_evolution.shape[1]
#     # fig, axes = plt.subplots(n_variables + 3, 1, figsize=(8, 6))
#     fig, axes = plt.subplots( 4, 1, figsize=(8, 6))
#     # fig, axes = plt.subplots(n_variables + 3, 1, figsize=(6, 1.5 * (n_variables + 3)))

#     # for i in range(n_variables//5):
#     #     axes[i].plot(range(n_generations + 1), variable_evolution[:, i, :], marker='o')
#     #     axes[i].set_xlabel('Generation')
#     #     axes[i].set_ylabel(f'Variable {i}')
#     #     axes[i].set_ylim(0, 1)  # Adjust the y-axis limits if needed

#     # axes[n_variables].plot(range(n_generations), angle_evolution, marker='o')
#     # axes[n_variables].set_xlabel('Generation')
#     # axes[n_variables].set_ylabel('Angle')

#     # axes[n_variables + 1].plot(range(n_generations), entropy_evolution, marker='o')
#     # axes[n_variables + 1].set_xlabel('Generation')
#     # axes[n_variables + 1].set_ylabel('Entropy')

#     # axes[n_variables + 2].plot(range(n_generations), density_evolution, marker='o')
#     # axes[n_variables + 2].set_xlabel('Generation')
#     # axes[n_variables + 2].set_ylabel('Density')
#     # ##print(n_variables)
#     # raise ValueError()

#     # axes[0].plot(range(n_generations), angle_evolution, marker='o')
#     # axes[0].imshow(angle_evolution[:n_generations].T, aspect='auto', cmap='hot', origin='lower')
#     # axes[0].bar(range(n_generations), angle_evolution)
#     axes[0].violinplot(angle_evolution.T, showmedians=True)
#     axes[0].set_xlabel('Generation')
#     axes[0].set_ylabel('Angle')
#     ##print(len(angle_evolution))

#     # axes[1].violinplot(entropy_evolution.T, showmedians=True)
#     axes[1].imshow(entropy_evolution[:n_generations].T, aspect='auto', cmap='hot', origin='lower')
#     # axes[1].plot(range(n_generations), entropy_evolution, marker='o')
#     axes[1].set_xlabel('Generation')
#     axes[1].set_ylabel('Entropy')

#     # axes[2].imshow(density_evolution, aspect='auto', cmap='hot', origin='lower')
#     axes[2].plot(range(n_generations), density_evolution, marker='o')
#     # axes[2].boxplot(density_evolution.T)
#     axes[ 2].set_xlabel('Generation')
#     axes[2].set_ylabel('Density')

#     ##print(cost_evolution)
#     # axes[3].plot(range(n_generations + 1), cost_evolution, marker='o')
#     # axes[3].boxplot(cost_evolution.T)
#     # axes[3].imshow(cost_evolution.reshape(1, -1), aspect='auto', cmap='hot', origin='lower')
#     axes[3].bar(range(n_generations), cost_evolution)
#     axes[3].set_xlabel('Generation')
#     axes[3].set_ylabel('Cost')        



#     plt.tight_layout()
#     plt.legend()
#     plt.show()
#     plt.clf()





""" functions paper Portfolio Selection via Particle Swam Optimization: state-of-the-art """
def plot_vars_auto(x,y1,y2,y3,y4,y5,local_min_func_df,glob_min_func_df,rule_r_p_actual_values_df,rule_r_p_df,num_gen_str,name_of_file='reactive',pareto=False):
    # import matplotlib.pyplot as plt

    # Sample data
    
    # ##print(local_min_func_df)
    # raise ValueError
    


    #print('global func ',glob_min_func_df)
    # raise ValueError
    maen_global=glob_min_func_df.values.tolist()[0]
    # #print(f'mean global >>>>>> {type(maen_global),glob_min_func_df,maen_global}')
    # raise ValueError
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
    # #print(maen_global)
    # #print(mean_local[0][0])
    # #print(mean_local)
    # raise ValueError
    smallest=mean_local[0]
    for v in mean_local:
        # #print(v[0])
        if v <smallest:
            local_only_lower.append(v)
            smallest=v
        else:
            local_only_lower.append(smallest)

    smallest=mean_r_p_panalty_only_lower[0][0]
    for v in mean_r_p_panalty_only_lower:
        if v[0]<smallest:
            r_p_penalty_only_lower.append(v[0])
            smallest=v[0]
        else:
            r_p_penalty_only_lower.append(smallest)


    greatest=mean_r_p_actual_values[0][0]
    for v in mean_r_p_actual_values:
        if v[0]>greatest:
            r_p_actual_values_only_lower.append(v[0])
            greatest=v[0]
        else:
            r_p_actual_values_only_lower.append(greatest)
    



    df_ar=rule_r_p_actual_values_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            if isinstance(df_ar[row,col],list):
                df_ar[row,col]=df_ar[row,col][0]


    # Find the maximum of this data frame and print it to
    rule_r_p_actual_values_df=pd.DataFrame(df_ar)
    print ("max of something")

    print(rule_r_p_actual_values_df.max())
    print ("\n")





    df_ar=local_min_func_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            if isinstance(df_ar[row,col],list):
                df_ar[row,col]=df_ar[row,col][0]



    local_min_func_df=pd.DataFrame(df_ar)


    # ##print(len(local_min_func_df.columns))
    # raise ValueError
    df_ar_g=glob_min_func_df.to_numpy()

    for row in range(df_ar_g.shape[0]):
        for col in range(df_ar_g.shape[1]):
            if isinstance(df_ar_g[row,col],list):
                df_ar_g[row,col]=df_ar_g[row,col][0]
    #print(df_ar_g)
    # ##print(np.array_equal(df_ar,df_ar_g))
    # raise ValueError

    glob_min_func_df=pd.DataFrame(df_ar_g)
    #print('final glob' ,glob_min_func_df)
    # raise ValueError

    df_ar=rule_r_p_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            if isinstance(df_ar[row,col],list):
                df_ar[row,col]=df_ar[row,col][0]


    rule_r_p_df=pd.DataFrame(df_ar)
    ##print(rule_r_p_df.equals(local_min_func_df))
    ##print(rule_r_p_df.equals(glob_min_func_df))
    ##print(local_min_func_df.equals(glob_min_func_df))
    # ##print(rule_r_p_df.equals(local_min_func_df))
    # raise ValueError
  # raise ValueError/Users/abdulsalamalmohamad/Documents/project 5/graphs/gt2sa/graphs_fixed/data_final
    import os

    # Specify the path and name of the directory
    # directory = '/Users/abdulsalamalmohamad/Documents/project 5/graphs/gt2sa/graphs_fixed/ /'+name_of_file+"_"+num_gen_str+'/'

    # /Users/abdulsalamalmohamad/Documents/project 5/graphs/all_reactiv4
    
    # directory = '/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"_"+num_gen_str+'/'
    if pareto:
        ##print(f'pareto is >>>>>>>>> {pareto}')
        directory = f'/Users/abnerandreymartinezzamudio/Downloads/BSP3/BSP3/scripts/version/scriptM/results/R_{R}_RATIO_MULTI{ratio_multiple}_h{h}_r_c{r_c}_r_p_multi{r_p_multiple}_cash_{total_cash}'+ 'with_pareto'+"_"+num_gen_str+'/'
    else:

        directory = f'/Users/abnerandreymartinezzamudio/Downloads/BSP3/BSP3/scripts/version/scriptM/results/R_{R}_RATIO_MULTI{ratio_multiple}_h{h}_r_c{r_c}_r_p_multi{r_p_multiple}'+"_"+num_gen_str+'/'
    # /Users/abdulsalamalmohamad/Documents/project 5/graphs/gt2sa/graphs_fixed/data_final
    # Create the directory
    os.mkdir(directory)


    # Check if the directory was created successfullys
    if os.path.exists(directory):
        print("Directory created successfully.")
    else:
        print("Failed to create the directory.")




    global df_r_p_porti
    df_r_p_porti.to_excel(directory+f'r_p_porti.xlsx', index=False)

    global df_risk_porti
    df_risk_porti.to_excel(directory+f'Risk_porti.xlsx', index=False)

    for i, row in local_min_func_df.iterrows():
        plt.plot(local_min_func_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('LOCAL MIN_FUNC EVOLUTION OVER Iterations')

    #     # Add legend
    #     plt.legend()

    #     # Display the plot
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"_"+num_gen_str+'/graphs_fixed/LOCAL_MIN_FUNC.png', dpi=1000)
    plt.legend()

    plt.savefig(directory+f' pareto___{pareto}____gen_LOCAL_MIN_FUNC_.png', dpi=500)
    plt.clf()
    
    global df_constraints
    for column in df_constraints.columns:
        plt.plot(df_constraints[column], label=column)
    # for i, row in df_constraints.iterrows():
    #     plt.plot(df_constraints.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('constraints_values  ')
        plt.title('constraints_values EVOLUTION OVER Iterations')

    #     # Add legend
    #     plt.legend()

    #     # Display the plot
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"_"+num_gen_str+'/graphs_fixed/LOCAL_MIN_FUNC.png', dpi=1000)
    plt.legend()

    plt.savefig(directory+f' pareto___{pareto}____constraints_values_.png', dpi=500)
    plt.clf()







    for i, row in rule_r_p_actual_values_df.iterrows():
        plt.plot(rule_r_p_actual_values_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('Rule R_P_Actual Values ')
        plt.title('Rule R_P Actual Values')

    #     # Add legend
    #     plt.legend()

    #     # Display the plot
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"_"+num_gen_str+'/graphs_fixed/LOCAL_MIN_FUNC.png', dpi=1000)
    plt.legend()
    # R=rendimento_medio_set_1.mean()*30
    # ratio_multiple=0.5
    # h=0.06
    # r_c=0.07
    # r_p_multiple=1.5
    plt.savefig(directory+f' pareto___{pareto}_Rule_R_p_actual_values_.png', dpi=500)
    plt.clf()
    
    # global_only_lower=[]
    # local_only_lower=[]
    # r_p_penalty_only_lower=[]
    # r_p_actual_values_only_lower=[]

    
    x_glob_only_lower=[i for i in range(len(global_only_lower))]
    x_local_only_lower=[i for i in range(len(local_only_lower))]
    x_r_p_penalty_only_lower=[i for i in range(len(r_p_penalty_only_lower))]
    x_r_p_actual_values_only_lower=[i for i in range(len(r_p_actual_values_only_lower))]

    #print(len(x_glob_only_lower), len(global_only_lower),x_glob_only_lower, global_only_lower)
    plt.xlabel('ITERATIONS')
    plt.ylabel('GLOBAL MIN_FUNC VALUE ')
    plt.title(' GLOBA LMIN_FUNC EVOLUTION STRICTLY DECREASING')
    plt.plot(x_glob_only_lower, global_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_GLOBAL_MIN_FUNC_EVOLUTION_STRICTLY_DEACREASING.png', dpi=500)
    plt.clf()





    plt.xlabel('ITERATIONS')
    plt.ylabel('LOCAL MIN_FUNC VALUE ')
    plt.title(' LOCAL_MIN_FUNC EVOLUTION STRICTLY DECREASING')
    plt.plot(x_local_only_lower, local_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_LOCAL_MIN_FUNC_EVOLUTION_STRICTLY_DEACREASING.png', dpi=500)
    plt.clf()



    plt.xlabel('ITERATIONS')
    plt.ylabel('R_P_ACTUAL_VALUE ')
    plt.title(' R_P ACTUAL EVOLUTION STRICTLY INCREASING')
    plt.plot(x_r_p_actual_values_only_lower, r_p_actual_values_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_R_P_ACTUAL_EVOLUTION_STRICTLY_INCREASING.png', dpi=500)
    plt.clf()



    plt.xlabel('ITERATIONS')
    plt.ylabel('R_P_PENALTY_VALUE ')
    plt.title(' R_P PENALTY EVOLUTION STRICTLY DECREASING')
    plt.plot(x_r_p_penalty_only_lower, r_p_penalty_only_lower)
    plt.savefig(directory+f' pareto___{pareto}_R_P_PENALTY_EVOLUTION_STRICTLY_DEACREASING.png', dpi=500)
    plt.clf()


    # plt.show()
    # raise ValueError
    # local_min_func_df_trans = local_min_func_df.transpose()

    
    # for i, row in local_min_func_df.iterrows():
    #     x = local_min_func_df.columns[1:]  # X-axis values
    #     y = row[1:]  # Y-axis values
    #     ##print("x is : ",x,'\n y is ',y,'\n')
    # raise ValueError
        # plt.plot(x, y, label=f'Row {i+1}')

# Add labels and title
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Lines for Each Row')
    # plt.show()
    # ##print(glob_min_func_df)
    # ##print(type(glob_min_func_df))
    # raise ValueError
    for i, row in glob_min_func_df.iterrows():
        # #print("row is ",row)
        plt.plot(glob_min_func_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('GLOBAL MIN_FUNC VALUES ')
        plt.title(' GLOBA LMIN_FUNC EVOLUTION FOR ALL SOLUTIONS')
    # raise ValueError
        # Add legend
        # plt.legend()

        # Display the plot
    # plt.show()
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"/"+num_gen_str+'/graphs_fixed/GLOBAL_MIN_FUNC.png', dpi=1000)
    plt.legend()
    plt.savefig(directory+f' pareto_{pareto}_GLOBAL_MIN_FUNC.png', dpi=500)
    plt.clf()
    for i, row in rule_r_p_df.iterrows():
        plt.plot(rule_r_p_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('R_p VALUES ')
        plt.title(' R_p  PENALTY EVOLUTION FOR ALL SOLUTIONS')

        # Add legend
        # plt.legend()

        # Display the plot
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"/"+num_gen_str+'/graphs_fixed/R_p.png', dpi=1000)
    
    plt.legend()
    plt.savefig(directory+f'_{pareto}_R_p_PENALTY.png', dpi=500)
    plt.clf()
    # plt.show()

    # Plotting
    # plt.plot(x, y1, label='rule_R')
    x_rp=[i for i in range(len(R_P_mean_list))]

    plt.plot(x_rp, R_P_mean_list, label='MEAN_R_P')
    # y_std=R_P_std_lis
    # plt.plot(x_rp, R_P_std_list, label='STANDARD_DEVIATION_R_P')
    plt.legend()
    # std=
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"/"+num_gen_str+'/graphs_fixed/MEAN_R_p.png', dpi=1000)
    plt.savefig(directory+f'__{pareto}___MEAN_STD_R_p.png', dpi=500)
    plt.clf()




    y_glob=glob_min_func_df.mean().to_list()
    
    x_glob=[i for i in range(len(y_glob))]
    plt.plot(x_glob, y_glob, label='MEAN_GLOB_FUNC')
    y_std=glob_min_func_df.std().to_list()
    # plt.plot(x_glob, y_std, label='STD_GLOB_FUNC')
    # plt.legend()
    ##print(glob_min_func_df,y_glob,y_std)
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"/"+num_gen_str+'/graphs_fixed/MEAN_GLOB_FUNC.png', dpi=1000)
    plt.legend()
    plt.savefig(directory+f'pareto___{pareto}____MEAN_STD_GLOB_FUNC.png', dpi=500)
    plt.clf()


    y_loc=local_min_func_df.mean().to_list()
    # ##print(local_min_func_df,y_glob)
    x_loc=[i for i in range(len(y_loc))]
    plt.plot(x_loc, y_loc, label='MEAN_LOC_FUNC')
    # y_std=local_min_func_df.std().to_list()
    plt.plot(x_loc, y_std, label='STD_LOC_FUNC')
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"/"+num_gen_str+'/graphs_fixed/MEAN_LOC_FUN.png', dpi=1000)
    plt.legend()
    plt.savefig(directory+f'pareto____{pareto}_____MEAN_STD_LOC_FUN.png', dpi=1000)
    plt.clf()

    plt.plot(x, y1, label='rule_R_p')
    plt.plot(x, y2, label='quantity_l_s')
    plt.plot(x, y3, label='card_l_s')
    plt.plot(x, y4, label='l_s_total_values')
    plt.plot(x, y5, label='us_reg')

    # Add labels and title
    plt.xlabel('ITERATIONS')
    plt.ylabel('VALUES')
    plt.title('PARAMETERS ')

    # Add legend
    plt.legend()

    # Display the plot
    # plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/'+name_of_file+"/"+num_gen_str+'/graphs_fixed/Parameters.png', dpi=1000)
    plt.savefig(directory+f'___{pareto}____Parameters.png', dpi=1000)
    plt.clf()
    # plt.show()

def plot_vars(x,y1,y2,y3,y4,y5,local_min_func_df,glob_min_func_df,rule_r_p_df,rule_r_p_actual_values_df):
    # import matplotlib.pyplot as plt

    # Sample data
    
    # ##print(local_min_func_df)
    # raise ValueError




    df_ar=rule_r_p_actual_values_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]



    rule_r_p_actual_values_df=pd.DataFrame(df_ar)
    for i, row in rule_r_p_actual_values_df.iterrows():
        plt.plot(rule_r_p_actual_values_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('Actual_R_P_VALUES EVOLUTION FOR ALL SOLUTIONS')

    #     # Add legend
    #     plt.legend()

    #     # Display the plot
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/RULE_R_P_ACTUAL_VALUES.png', dpi=1000)
    plt.clf()





    df_ar=local_min_func_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]



    local_min_func_df=pd.DataFrame(df_ar)


    # ##print(len(local_min_func_df.columns))
    # raise ValueError
    df_ar_g=glob_min_func_df.to_numpy()
    for row in range(df_ar_g.shape[0]):
        for col in range(df_ar_g.shape[1]):
            df_ar_g[row,col]=df_ar_g[row,col][0]

    # ##print(np.array_equal(df_ar,df_ar_g))
    # raise ValueError

    glob_min_func_df=pd.DataFrame(df_ar_g)


    df_ar=rule_r_p_df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]



    rule_r_p_df=pd.DataFrame(df_ar)
    ##print(rule_r_p_df.equals(local_min_func_df))
    ##print(rule_r_p_df.equals(glob_min_func_df))
    ##print(local_min_func_df.equals(glob_min_func_df))
    # ##print(rule_R_p_df.equals(local_min_func_df))
    # raise ValueError

    # for index, row in new_test_fd.iterrows():
    #     ##print(row)

    # raise ValueError
    # Iterate through each cell and change values
    # for index, row in local_min_func_df.iterrows():
    #     for column in local_min_func_df.columns:
    #         cell_value = row[column]
    #         # Modify the cell value
    #         # new_value = cell_value * 2
    #         # Assign the new value to the cell
    #         local_min_func_df.at[index, column] = cell_value[0]

    # ##print(local_min_func_df)
    # ##print(local_min_func_df)
    # raise ValueError
    # Iterate through each cell and change values
    # for index, row in glob_min_func_df.iterrows():
    #     for column in glob_min_func_df.columns:
    #         cell_value = row[column]
    #         # Modify the cell value
    #         # new_value = cell_value * 2
    #         # Assign the new value to the cell
    #         glob_min_func_df.at[index, column] = cell_value[0]

    
    # Iterate through each cell and change values
    # for index, row in rule_R_p_df.iterrows():
    #     ##print(row)
    #     # for column in rule_R_p_df.columns:
    #     #     cell_value = row[column]
    #     #     # Modify the cell value
    #     #     # new_value = cell_value * 2
    #     #     # Assign the new value to the cell
    #     #     rule_R_p_df.at[index, column] = cell_value[0]
    #     #     ##print(f'old value is {cell_value} new value {row[column]}')
    # raise ValueError
    # ##print(glob_min_func_df)
    # raise ValueError
    for i, row in local_min_func_df.iterrows():
        plt.plot(local_min_func_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('LOCAL MIN_FUNC VALUES ')
        plt.title('LOCAL MIN_FUNC EVOLUTION FOR ALL SOLUTIONS')

    #     # Add legend
    #     plt.legend()

    #     # Display the plot
    # plt.ylim(0, 0.6)  # Set the y-axis limits as per your desired range
    # plt.yticks([i/100 for i in range(7)])
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/LOCAL_MIN_FUNC.png', dpi=1000)
    plt.clf()
    # plt.show()
    # raise ValueError
    # local_min_func_df_trans = local_min_func_df.transpose()

    
    # for i, row in local_min_func_df.iterrows():
    #     x = local_min_func_df.columns[1:]  # X-axis values
    #     y = row[1:]  # Y-axis values
    #     ##print("x is : ",x,'\n y is ',y,'\n')
    # raise ValueError
        # plt.plot(x, y, label=f'Row {i+1}')

# Add labels and title
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Lines for Each Row')
    # plt.show()

    for i, row in glob_min_func_df.iterrows():
        plt.plot(glob_min_func_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('GLOBAL MIN_FUNC VALUES ')
        plt.title(' GLOBA LMIN_FUNC EVOLUTION FOR ALL SOLUTIONS')

        # Add legend
        # plt.legend()

        # Display the plot
    # plt.show()
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/GLOBAL_MIN_FUNC.png', dpi=1000)
    plt.clf()
    for i, row in rule_r_p_df.iterrows():
        plt.plot(rule_r_p_df.columns[1:], row[1:])

        # Add labels and title
        plt.xlabel('ITERATIONS')
        plt.ylabel('R_p VALUES ')
        plt.title(' R_p EVOLUTION FOR ALL SOLUTIONS')

        # Add legend
        # plt.legend()

        # Display the plot
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/R_p.png', dpi=1000)
    plt.clf()
    # plt.show()

    # Plotting
    # plt.plot(x, y1, label='rule_R')
    x_rp=[i for i in range(len(R_P_mean_list))]
    plt.plot(x_rp, R_P_mean_list, label='MEAN_R_P')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/MEAN_R_p.png', dpi=1000)
    plt.clf()


    y_glob=glob_min_func_df.mean().to_list()
    ##print(glob_min_func_df,y_glob)
    x_glob=[i for i in range(len(y_glob))]
    plt.plot(x_glob, y_glob, label='MEAN_GLOB_FUNC')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/MEAN_GLOB_FUNC.png', dpi=1000)
    plt.clf()

    y_loc=local_min_func_df.mean().to_list()
    # ##print(local_min_func_df,y_glob)
    x_loc=[i for i in range(len(y_loc))]
    plt.plot(x_loc, y_loc, label='MEAN_GLOB_FUNC')
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/MEAN_LOC_FUNC.png', dpi=1000)
    plt.clf()

    plt.plot(x, y1, label='rule_r_p')
    plt.plot(x, y2, label='quantity_l_s')
    plt.plot(x, y3, label='card_l_s')
    plt.plot(x, y4, label='l_s_total_values')
    plt.plot(x, y5, label='us_reg')

    # Add labels and title
    plt.xlabel('ITERATIONS')
    plt.ylabel('VALUES')
    plt.title('PARAMETERS ')

    # Add legend
    plt.legend()

    # Display the plot
    plt.savefig('/Users/abdulsalamalmohamad/Documents/project 5/graphs/graphs_fixed/parameters.png', dpi=1000)
    plt.clf()
    # plt.show()
    #print(f'\n\n\n R is :   {R}')


def get_a_value_df_col(df,col,var):
    # for index in range(len(df)):
    
    # df_df=df.to_frame()
    #print(df.columns)
    df.columns=['Variables', 'Effect', 'Cumulative Percentage', 'Ratio']
    # variables_column = df
    if isinstance(df, pd.DataFrame):
        variables_column = df
    elif isinstance(df, pd.Series):
    # else:
        variables_column=variables_column.to_frame()
    # #print(type(variables_column))
    # for index, row in variables_column.items():
    for index,row in variables_column['Variables'].items():
        # Access the value using row
        if row == var:
            req_value = variables_column[col][index]
            return req_value
    # if isinstance(df, pd.Series):
    #     # #print(f'Variable is a Pandas Series')
    #     #print("Variable is a Pandas Series")
    #     df_df=df.to_frame()

    # else:
    #     #print("Variable is not a Pandas Series")
    #     df_df=deepcopy(df)
    #     #print(type(df_df))
    # # raise ValueError
    
    # #print(type(df_df.dtypes))
    # for index,row in df_df['Variables'].iteritems():
    #     # ##print(df['Variables'])
    #     # raise ValueError
    #     if row==var:
    #         req_value=df[col][index]
    #         return req_value
    raise ValueError
def pareto_analysis(values,desired_cumulative_percentage):

    # ratio_multiple=0.01
    # Define the variables and their effects
    # values=[rule_R_p_scalar,quantity_l_s_scalar,card_l_s_scalar,l_s_total_values_scalar,us_reg_scalar]
    # #print(values)
    # raise ValueError
    global r_p_multiple
    global ratio_multiple
    data = {
        'Variables': [0,1,2,3,4],
        'Effect': values
    }
    # #print(f'\values are {values}')
    # raise ValueError

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Sort the DataFrame by effect in descending order
    df = df.sort_values(by='Effect', ascending=False)

    # Calculate cumulative percentage
    total_effect = df['Effect'].sum()
    df['Cumulative Percentage'] = (df['Effect'].cumsum() / total_effect) * 100
    #print(total_effect)
    #print(df['Effect'].cumsum())
    #print(df['Cumulative Percentage'])
    desired_cumulative_percentage=df['Cumulative Percentage'].mean()
    # raise ValueError
    # Set the desired cumulative percentage
    # desired_cumulative_percentage = 80

    # #print(df)
    # Identify the variables that contribute to the desired cumulative percentage
    selected_variables = df[df['Cumulative Percentage'] <= desired_cumulative_percentage ]
    #print(f'selected vairables are {selected_variables}')
    # raise ValueError

    selected_variables_list=selected_variables['Variables'].to_list()
    # ##print(selected_variables,selected_variables_list)
    # raise ValueError
    unselected_vars=[]
    for index in range(len(df)):
        if df['Variables'][index] not in selected_variables_list:
            unselected_vars.append(df['Variables'][index] )

    


    
    # ##print(selected_variables,selected_variables_list,unselected_vars)

    # variables=[rule_R_p_scalar,quantity_l_s_scalar,card_l_s_scalar,l_s_total_values_scalar,us_reg_scalar]


    # Calculate the ratios of the selected variables' effects to the total effect
   
    total_selected_effect = sum(values)
    selected_variables['Ratio']=['' for x in range(len(selected_variables))]
    # selected_variables.insert(len(selected_variables), "Ratio", ["" for _ in list(range(len(selected_variables)))], True)
    # ##print(selected_variables)
    
    # raise ValueError
    selected_variables['Ratio'] = selected_variables['Effect'] / total_selected_effect
    
    
    for selected_var in selected_variables_list:
        
        var_ratio=get_a_value_df_col(selected_variables,'Ratio',selected_var)
        

        if selected_var==1:
            global quantity_l_s_scalar
            quantity_l_s_scalar=quantity_l_s_scalar*var_ratio*ratio_multiple
            # quantity_l_s_scalar=quantity_l_s_scalar*var_ratio
            # global quantity_l_s_scalar_values
            # quantity_l_s_scalar_values.append(quantity_l_s_scalar)

        elif selected_var==2:
            
            global card_l_s_scalar
            card_l_s_scalar=card_l_s_scalar*var_ratio*ratio_multiple
            # card_l_s_scalar=card_l_s_scalar*var_ratio
            # global card_l_s_scalar_values
            # card_l_s_scalar_values.append(card_l_s_scalar)

        

        elif selected_var==3:
            
            global l_s_total_values_scalar
            l_s_total_values_scalar=l_s_total_values_scalar*var_ratio*ratio_multiple
            # global l_s_total_values_scalar_values
            # l_s_total_values_scalar_values.append(l_s_total_values_scalar)
        
        elif selected_var==4:
            global us_reg_scalar
            us_reg_scalar=us_reg_scalar*var_ratio*ratio_multiple
            # global us_reg_scalar_values
            # us_reg_scalar_values.append(us_reg_scalar)
            # for unsel_var in unselected_vars:

    global rule_R_p_scalar_values
    rule_R_p_scalar_values.append(rule_R_p_scalar)
    
    # global rule_R_p_values
    # rule_R_p_values.append(rule_R_p_scalar_values)
    global quantity_l_s_scalar_values
    quantity_l_s_scalar_values.append(quantity_l_s_scalar)

    global card_l_s_scalar_values
    card_l_s_scalar_values.append(card_l_s_scalar)

    global l_s_total_values_scalar_values
    l_s_total_values_scalar_values.append(l_s_total_values_scalar)

    global us_reg_scalar_values
    us_reg_scalar_values.append(us_reg_scalar)

    new_values=[rule_R_p_scalar,quantity_l_s_scalar,card_l_s_scalar,l_s_total_values_scalar,us_reg_scalar]
    # global ratio_multiple
    # ratio_multiple=ratio_multiple*0.25

    # Output the selected variables and their ratios
    ##print("Selected Variables and Ratios:")
    ##print(selected_variables[['Variables', 'Ratio']])
    ##print(f'\n\n\n\n\n Values are {new_values}\n\n\n')


def const_eight_nine(pop,R,scalar_8=scalar_8,scalar_9=scalar_9):
    
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
                # solutions_quantity[solution,index]=abs(asset-min_quant)
            
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
                # break
        
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
            # ##print(x_i)
            # ##print(abs_pop[solution][index])
            # raise ValueError
        ret_asset_minus_r=ret_times_asset-R
        if ret_asset_minus_r<0:

            fitness_credit[solution]=abs(ret_asset_minus_r)*scalar_1
    # ##print(fitness_credit-fitness_credit)   
    # raise ValueError

            # abs_pop[solution,index]=abs(abs_pop[solution,index])

    return fitness_credit



def rule_2(pop,scalar_2=scalar_2):
    fitness_credit=np.zeros((pop.shape[0],1))
    
    # abs_pop=deepcopy(pop)
    for solution in range(pop.shape[0]):
        sum_asset=0
        for index in range(pop.shape[1]-1):
            sum_asset+=pop[solution,index]

        sum_asset_minus_1=sum_asset-1
        if sum_asset_minus_1!=0:

            fitness_credit[solution]=abs(sum_asset_minus_1)*scalar_2
        # fitness_credit[solution]=1-sum_asset
    # ##print('credit r_2',fitness_credit)
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
            # ##print('\n\n\n x_cash,sum_min',x_cash,sum_min)

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
                # h_i=0.2
                h_i=h
            # r_c=h_i-pop[solution,-1]
            # r_c=0.01
            r_i=rendimento_medio_set_1[index]
            sum_assets+=((r_i-h_i)*r_c)*asset
        if sum_assets<R:
            credit_18[solution]=abs(sum_assets-R)*scalar_18
    return credit_18

def rule_twelve(pop,scalar_12=scalar_12):
    fitness_credit=np.zeros((pop.shape[0],1))
    
    # abs_pop=deepcopy(pop)
    for solution in range(pop.shape[0]):
        sum_asset=0
        for index in range(pop.shape[1]-1):
            sum_asset+=abs(pop[solution,index])

        sum_asset_minus_2=sum_asset-2
        if sum_asset_minus_2>0:

            fitness_credit[solution]=abs(sum_asset_minus_2) * scalar_12
        # fitness_credit[solution]=1-sum_asset
    # ##print('credit r_2',fitness_credit)
    return fitness_credit

""" Here is the definition of the objective function

paper: Portfolio Selection via Particle Swam Optimization: state-of-the-art ......  """

def new_objective_function(pop,R,min_quant,max_quant,cardin_min,cardin_max):
    pop_without_cash=deepcopy(pop)
    # pop_without_cash=pop_without_cash.T
    pop_without_cash=pop_without_cash[:,:-1]
    fitness=np.zeros((pop.shape[0],1))
    # weights=deepcopy(pop)

    for solution in range(pop.shape[0]):
        local_fitness=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            for other_index,other_asset in enumerate(pop_without_cash[solution,:]):
                # ##print('cov',matrice_correlazione_1[index][other_index])
                # raise ValueError
                local_fitness+=matrice_correlazione_1[index][other_index]*asset*other_asset
                #objective function 
                # ##print(matrice_correlazione_1[index][other_index])
                # ##print(matrice_correlazione_1[index,other_index])
                # raise ValueError
                # ##print(local_fitness)
        fitness[solution]=local_fitness
    # Portfolio Selection via Particle Swam Optimization: state-of-the-art ...... 1.0
    ##print('min function Paper 1 is  >>>>>>>>>',fitness)
    # ##print('fitness shape',fitness.shape)
    # ##print('fitness before',fitness)
    # raise ValueError

    # Portfolio Selection via Particle Swam Optimization: state-of-the-art ...... 1.1
    r_1=rule_one(pop,R,scalar_1)
    ##print('r_1 paper 1 is ',r_1)
    # ##print('r_1 is ',r_1)
    # raise ValueError
    
    # ##print(np.equal())
    fitness=fitness+r_1
    # ##print('fitness after',fitness)
    
    # raise ValueError
    # Portfolio Selection via Particle Swam Optimization: state-of-the-art ...... 2
    r_2=rule_2(pop,scalar_2)
    ##print('r_2 paper 1 is ',r_2)
    
    fitness=fitness+r_2
    # ##print('fitness before',fitness)

    # Portfolio Selection via Particle Swam Optimization: state-of-the-art ...... 5
    quantity_credit=quantity(pop,min_quant,max_quant,scalar_quant)
    ##print('quantity_credit paper 1 is ',quantity_credit)
    fitness=fitness+quantity_credit
    # ##print('fitness before is >>>>',fitness)
    # Portfolio Selection via Particle Swam Optimization: state-of-the-art ...... 4
    cardinality_credit=cardinality(pop,cardin_min,cardin_max,scalar_cardinality)
    ##print('cardinality_credit paper 1 is ',cardinality_credit)
    fitness=fitness+cardinality_credit




    

    # fitness=fitness+cardinality_credit
    # ##print('fitness before eight',fitness)

    # Portfolio Selection via Particle Swam Optimization: state-of-the-art ...... 8 and 9
    eight,nine=const_eight_nine(pop,R,scalar_8,scalar_9)
    ##print('r_8 paper 1 is ',eight)
    ##print('r_9 paper 1 is ',nine)
    # ##print(eight)
    # ##print(nine)
    # raise ValueError
    fitness=fitness+eight
    # ##print('fitness after eight',fitness)

    fitness=fitness+nine

    twelve=rule_twelve(pop,scalar_12)
    ##print('r_12 paper 1 is ',twelve)
    fitness=fitness+twelve
    r_17=rule_17(pop)
    ##print('r_17 paper 1 is ',r_17)
    fitness=fitness+r_17
    r_18= rule_18(pop)
    ##print('r_18 paper 1 is ',r_18)
    fitness=fitness+r_18

    """ # Portfolio Selection via Particle Swam Optimization: state-of-the-art 
    here just turning it into suitable object type (timeseries)"""
    fit_ser=[]
    for solution in range(pop.shape[0]):
        # ##print(fitness[solution])
        fit_ser.append(fitness[solution])
    # fit_ser=np.array(fit_ser)
    # ##print("final fitness",fitness)
    fitness=pd.Series(fit_ser)

    for index,elemnt in enumerate(fitness):
        # ##print('element is ',elemnt)
        fitness[index]=elemnt[0]
        # ##print('element is ',obiettivo[index])
    # ##print('objective function is ',fitness)
    # ##print(fitness)
    # raise ValueError
    # ##print('fitness after nine',fitness)
    # ##print('\n\n >>>>',fitness[0]==fitness_2[0])
    # ##print(np.equal(fitness_2,fitness))
    # raise ValueError
    """ # Portfolio Selection via Particle Swam Optimization: state-of-the-art 
    
    here we return the call of the object function times -1 because the whole code is trying to maxmise 
    and so for each solution s the penalty given -s , maximise that and you get the minimum ,thats better than tracing 
    everywhere he maximised in one way or another at different stages of the execution  and chaning it """
    fitness=-fitness
    return fitness
# nuova_pop=np.random.uniform(low=-1, high=1, size=(20,32))
# ##print(nuova_pop.shape)
# R=rendimento_medio_set_1.mean()
# min_quant=-1
# max_quant=1
# cardin_min=nuova_pop.shape[1]//2
# cardin_max=nuova_pop.shape[1]

# ##print('R is :',R)

# rule_one(nuova_pop,R)
# ##print(nuova_pop)
# raise ValueError
# nuova_pop[0,:]=[0 for i in range(32)]
# ##print(cardinality(nuova_pop))
# ##print(quantity(nuova_pop,-0.5,0.5))
# ##print(const_eight(nuova_pop))
# const_nine(nuova_pop)
# const_eleven(nuova_pop)
# const_twelve(nuova_pop)
# new_objective_function(nuova_pop,R,min_quant,max_quant,cardin_min,cardin_max)

#end first paper





""" functions paper Genetic algorithm with short selling........ """
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

            # ##print(w_long[0,:])
            # raise ValueError
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
        global list_const
        list_const.append(quant_credit)
        
        # solutions_quantity[solution]=quant_credit
            
                
    return solutions_quantity


def cardinality_long_short(pop_without_cash,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar):
    solutions_cardinality=np.zeros((pop_without_cash.shape[0],1))
    # ##print(pop_without_cash.shape)
    # raise ValueError
    global list_const
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
            # ##print('cardinality_local',cardinality_local)
        if cardinality_local<cardin_min_l_s :
            solutions_cardinality[solution,:]=abs(cardinality_local-cardin_min_l_s)* card_l_s_scalar
            # solutions_cardinality[solution,:]=abs(cardinality_local-cardin_min_l_s)
            
            list_const.append(abs(cardinality_local-cardin_min_l_s))
        elif  cardinality_local>cardin_max_l_s:

            solutions_cardinality[solution,:]=abs(cardinality_local-cardin_max_l_s)* card_l_s_scalar
            # solutions_cardinality[solution,:]=abs(cardinality_local-cardin_max_l_s)
           
            list_const.append(abs(cardinality_local-cardin_max_l_s))
    return solutions_cardinality

import math
from decimal import Decimal, getcontext
# def rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar):

#     # r_p=0

#     r_p_penalty=np.zeros((pop_without_cash.shape[0],1))
#     r_p_actual=np.zeros((pop_without_cash.shape[0],1))

#     for solution in range(pop_without_cash.shape[0]):
        

#         term1=0
#         term2=0
#         term3=0
#         for index,asset in enumerate(pop_without_cash[solution,:]):
#             r_i=rendimento_medio_set_1[index]
#             term1+=(r_i*w_long[solution,index])

            
#     # for solution in range(pop_without_cash.shape[0]):
        

#             # term2=0
#         # for index,asset in enumerate(pop_without_cash[solution,:]):
#             # r_i=rendimento_medio_set_1[index]
#             term2+=(r_i*w_short[solution,index])


#     # for solution in range(pop_without_cash.shape[0]):
        

#             # term3=0
#         # for index,asset in enumerate(pop_without_cash[solution,:]):
#             # r_i=rendimento_medio_set_1[index]
#             if asset>0:
#                 h_i=0
#             else:
#                 h_i=h
#             term3+=(r_c*w_short[solution,index]*h_i)
        
#         #result = (float(R) - (float(term1) - float(term2) + float(term3))) * float(rule_R_p_scalar)


#         # from decimal import Decimal, getcontext

#         # Set the desired precision for Decimal calculations
#         getcontext().prec = 30

#         result = (Decimal(str(R)) - (Decimal(str(term1)) - Decimal(str(term2)) + Decimal(str(term3)))) * Decimal(str(rule_R_p_scalar))

#         if math.isinf((result)):
#             r_p_penalty[solution]=float('inf')
#         else:
#             r_p_penalty[solution]=result
#         # r_p_penalty[solution]=(R-(term1-term2+term3))
#         r_p_actual[solution]=int(term1-term2+term3)
#     return r_p_penalty,r_p_actual


def Long_short_total_values(pop_without_cash,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar):
    fitness_l_s=np.zeros((pop_without_cash.shape[0],1))
    # inverstors_value=1
    # abs_pop=deepcopy(pop)
    for solution in range(pop_without_cash.shape[0]):
        sum_w_l=0
        sum_w_s=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
            sum_w_l+=w_long[solution,index]
            sum_w_s+=w_short[solution,index]

        total_sum=sum_w_l-sum_w_s
        fitness_local=abs(total_sum-investor_value)
        ##print(f'\n\n\n\n\n\n\n fitness local {fitness_local}\n\n\n\n\n\n\n ')
        
        if fitness_local>T:
            fitness_l_s[solution]=(abs(fitness_local)-T)*l_s_total_values_scalar
            # fitness_l_s[solution]=(abs(fitness_local)-T)
            global list_const
            list_const.append((abs(fitness_local)-T))
        else:
            list_const.append(0)
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
        # fitness_local=abs(total_sum-investor_value)

        if total_sum>2:
            fitness_us_reg[solution]=abs(total_sum-2) * us_reg_scalar
            # fitness_us_reg[solution]=abs(total_sum-2) 
            global list_const
            list_const.append(abs(total_sum-2))
        else:
            list_const.append(0)
    return fitness_us_reg


# Genetic algorithm with short selling......objective function 
def new_objective_function_2(pop,R,min_quant,max_quant,cardin_min,cardin_max,pareto=False): 
    #Genetic algorithm with short selling .....2.1
    fitness=np.zeros((pop.shape[0],1))


    pop_without_cash=deepcopy(pop)
    # pop_without_cash=pop_without_cash.T
    pop_without_cash=pop_without_cash[:,:-1]

    new_xi=np.zeros((pop_without_cash.shape))
    # weights=deepcopy(pop)
    w_long=np.zeros((pop_without_cash.shape))
    w_short=np.zeros((pop_without_cash.shape))
    
    # ##print(pop_without_cash.shape)
    # raise ValueError
    df=pd.DataFrame({})

    for solution in range(pop_without_cash.shape[0]):
        local_fitness=0
        for index,asset in enumerate(pop_without_cash[solution,:]):
           
            
            
            new_xi[solution,index]=(abs(pop[solution,index]) * mean_price[index])/total_cash
            # ##print(new_xi[solution,index])
            if pop_without_cash[solution,index]>0:
                w_long[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash
            elif pop_without_cash[solution,index]<0:
                w_short[solution,index]=(abs(pop_without_cash[solution,index]) * mean_price[index])/total_cash

    # ##print(w_long[0])
    # raise ValueError

    
    for solution in range(pop_without_cash.shape[0]):
        sum_1=0
        for i in range((pop_without_cash.shape[1]*2)):
            if i>=(pop_without_cash.shape[1]):
                row=i-pop_without_cash.shape[1]
                w_i=w_short[solution,row]
            else:
                row=i
                w_i=w_long[solution,row]
            # w_i=new_xi[solution,row]

            for j in range((pop_without_cash.shape[1]*2)):  
                if j>=(pop_without_cash.shape[1]):
                    col=j-pop_without_cash.shape[1]
                    w_j=w_short[solution,col]
                else:
                    col=j
                    w_j=w_long[solution,col]

                # w_j=new_xi[solution,col]
                segma_i_j=matrice_correlazione_1[row][col]
                local_fitness=segma_i_j*w_i*w_j
                # ##print(w_i,w_j,segma_i_j,local_fitness)
                sum_1+=local_fitness
        # ##print(sum_1)
        #returns the variance of the portfolio without taking into account the constraint violations

        fitness[solution]=sum_1
    # Genetic algorithm with short selling......2.1 
    if len(fitness)==20:

        global local_min_func_df
        local_min_func_df.insert(len(local_min_func_df.columns), "", fitness.tolist(), True)
        # ##print(local_min_func_df.columns)
        ##print('>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n',local_min_func_df,'>>>>>>>>>>>>>>>>>>>>\n\n\n\n\n')
        # raise ValueError
    ##print("min_function Paper 2",fitness)
    # Genetic algorithm with short selling .....2.1 R_p
    # r_p_penalty,r_p_actual
    rule_rp,r_p_actual=rule_R_p(pop,pop_without_cash,new_xi,w_long,w_short,rule_R_p_scalar=rule_R_p_scalar)
    
    # global rule_r_p_actual_values
    # rule_r_p_actual_values.append(r_p_actual)
    

    
    # df.insert(2, "Age", [21, 23, 24, 21], True)
    # ##print('rule_r_p is ',rule_rp)
    if len(rule_rp)==20:
        global rule_r_p_df
        # df_ar_g=rule_rp.to_numpy()
        # for row in range(df_ar_g.shape[0]):
        #     for col in range(df_ar_g.shape[1]):
        #         if isinstance(df_ar_g[row,col],list):
        #             df_ar_g[row,col]=df_ar_g[row,col][0]
        # rule_nums=pd.DataFrame(df_ar_g)
        rule_r_p_df.insert(len(rule_r_p_df.columns), "", rule_rp.tolist(), True)
        df.insert(len(df.columns), "", rule_rp.tolist(), True)
        m=rule_rp.mean()
        global R_P_mean_list
        R_P_mean_list.append(m)
        global rule_r_p_actual_values_df
        rule_r_p_actual_values_df.insert(len(rule_r_p_actual_values_df), "", r_p_actual.tolist(), True)

        ##print(f"\n\n\n\n R_P_MEAN is >>>>>>>>>>>>>>>>>{m} \n\n\n\n ")
    fitness=fitness-rule_rp
    # ##print('fitness is : ',fitness)
    # raise ValueError
    # neg_rp=-rule_rp
    # ##print(rule_rp.shape)
    # df.insert(len(df), "", rule_rp.tolist(), True)
    

    # Genetic algorithm with short selling......2.3
    if len(quantity_l_s)==20:
        quantity_l_s=quantity_long_short(pop_without_cash,w_long,w_short,min_quant_long,max_quant_long,min_quant_short,max_quant_short,quantity_l_s_scalar=quantity_l_s_scalar)
        ##print('quantity_long_short Paper 2 is ',quantity_l_s)
        ##print(type(quantity_l_s),quantity_l_s.shape,len(df))
        df.insert(len(df), "", quantity_l_s.tolist(), True)
        # ##print(quantity_l_s)
        # ##print(df)
        # raise ValueError

    # raise ValueError
    fitness=fitness+quantity_l_s


    # Genetic algorithm with short selling......2.2
    if len(cardinality_l_s)==20:
        cardinality_l_s=cardinality_long_short(pop_without_cash,w_long,w_short,cardin_min_l_s,cardin_max_l_s,card_l_s_scalar=card_l_s_scalar)
        df.insert(len(df), "", cardinality_l_s.tolist(), True)
    ##print('cardinality_l_s Paper 2 is ',cardinality_l_s)
    fitness=fitness+cardinality_l_s
    # df.insert(2, "", cardinality_l_s.tolist(), True)


    # Genetic algorithm with short selling......2.6
    
    l_s_total_values=Long_short_total_values(pop_without_cash,w_long,w_short,l_s_total_values_scalar=l_s_total_values_scalar)
    if len(cardinality_l_s)==20:
        df.insert(len(df), "", l_s_total_values.tolist(), True)
    ##print('l_s_total_values Paper 2 is ',l_s_total_values)
    fitness=fitness+l_s_total_values
    # df.insert(3, "", l_s_total_values.tolist(), True)
    
    # Genetic algorithm with short selling......2.7 
    
    us_reg=usa_regulation(pop_without_cash,w_long,w_short,us_reg_scalar=us_reg_scalar)
    if len(cardinality_l_s)==20:
        df.insert(len(df), "", us_reg.tolist(), True)
    ##print('us_reg Paper 2 is ',us_reg)
    fitness=fitness+us_reg
    # df.insert(4, "", us_reg.tolist(), True)

    df_ar=df.to_numpy()
    for row in range(df_ar.shape[0]):
        for col in range(df_ar.shape[1]):
            df_ar[row,col]=df_ar[row,col][0]
    ##print(df)
    ##print(len(df.columns))
    ##print(df_ar.shape)
    # ##print(df_ar[0,:])
    # df=df[1:]
    # df.columns=[0,1,2,3,4]
    # ##print(df.columns )
    # df.drop(0, inplace=True, axis=0)
    ##print('>>>>>>>>',df)
    
    fitness_mean = df.mean(axis = 0, skipna = False).to_list()


    ##print('>>>>>>>> real mean',fitness_mean)
    # fitness_mean[0]=0
    
    # ##print('>>>>>>>> -R_p mean',fitness_mean)
    # ##print(len(fitness))
    # raise ValueError
    if pareto==True:

        pareto_analysis(fitness_mean,97)
    # raise ValueError


    # raise ValueError
    #overall cost function of paper :Genetic algorithm with short selling ...


    # ##print(len(fitness))
    # raise ValueError
    if len(fitness)==20:
        global glob_min_func_df
        glob_min_func_df.insert(len(glob_min_func_df.columns), "", fitness.tolist(), True)
        # ##print()
    fit_ser=[]
    for solution in range(pop.shape[0]):
        # ##print(fitness[solution])
        fit_ser.append(fitness[solution])
    # fit_ser=np.array(fit_ser)
    # ##print("final fitness",fitness)
    fitness=pd.Series(fit_ser)

    for index,elemnt in enumerate(fitness):
        # ##print('element is ',elemnt)
        fitness[index]=elemnt[0]
        # raise ValueError
    
    # Genetic algorithm with short selling......final fitness after all constraints
    
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
    #matrice_correlazione_1=rendimenti_set_1.corr()
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
            #fRP[:,j]=(((pesi[:,j]*Cx[j,:])/portvar)-(1/geni))**2
            #fRP[:,j]=(((pesi[:,j]*Cx[j,:])/portvar)-(1/geni))**2
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
    # ##print('length is' ,len(pop))
    # ##print("here is pop ",pop.T.shape)
    # ##print("here is return_matrix ",rendimenti_set_1.shape)
    # raise ValueError
    
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    # ##print('monthly_return_proti ',rendimento_portafoglio_mensile.shape)
    # ##print('min',np.minimum(rendimento_portafoglio_mensile,0).shape)
    # ##print('sum_shape',np.sum(np.minimum(rendimento_portafoglio_mensile,0),axis=0).shape)
    # ##print('sum',np.sum(np.minimum(rendimento_portafoglio_mensile,0),axis=0))
    # raise ValueError
    omega=(np.sum(np.minimum(rendimento_portafoglio_mensile,0),axis=0)/np.sum(np.maximum(rendimento_portafoglio_mensile,0),axis=0))
    ##print('omega is ',omega)
    # raise ValueError
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
        #costo=-contameno
    elif conta=='piu':
        contapiu=np.count_nonzero(rendimento_portafoglio_mensile>gain,axis=0)/rendimento_portafoglio_mensile
        costo=rendimento(pop)+delta*(contapiu)
    return costo
    

def twosided(pop):
    a=0.25
    pesi=pop
    rendimento_portafoglio_mensile=rendimenti_set_1@pesi.T
    twoside=np.zeros(len(rendimento_portafoglio_mensile.T))
    #upside=rendimento_portafoglio_mensile
    #downside=rendimento_portafoglio_mensile
    upside=np.maximum(rendimento_portafoglio_mensile-rendimento_portafoglio_mensile.mean(),0)
    downside=np.maximum(rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile,0)
##    for i in range(len(rendimento_portafoglio_mensile.T)):
##	    for j in range(len(rendimento_portafoglio_mensile)):
##		    if rendimento_portafoglio_mensile.iloc[j,i]-rendimento_portafoglio_mensile.iloc[:,i].mean()<0:
##			    upside.iloc[j,i]=0
##    for i in range(len(rendimento_portafoglio_mensile.T)):
##	    for j in range(len(rendimento_portafoglio_mensile)):
##		    if rendimento_portafoglio_mensile.iloc[:,i].mean()-rendimento_portafoglio_mensile.iloc[j,i]<0:
##			    downside.iloc[j,i]=0
    #upside=np.max((rendimento_portafoglio_mensile-rendimento_portafoglio_mensile.mean()),0)
    #downside=np.max((rendimento_portafoglio_mensile.mean()-rendimento_portafoglio_mensile),0)
    for z in range(len(rendimento_portafoglio_mensile.T)):
	    twoside[z]=-a*np.linalg.norm(upside.iloc[:,z],ord=1)-(1-a)*np.linalg.norm(downside.iloc[:,z],ord=2)+rendimento_portafoglio_mensile.mean()[z]
    #twosided=a*np.linalg.norm(upside,ord=1)+(1-a)*np.linalg(downside,ord=2)-rendimento_portafoglio_mensile
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

def minimax(pop):##minimax
    lambda_1=0.5
    minimax=np.zeros(len(pop))
    for i in range(len(pop)):
	#rendimento_portafoglio[i] = rendimento_medio_set_1 * pesi[i] * 12
	#np.sum(np.abs(rendimenti_set_1-*pesi[i],axis=1))
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
    #var=(norm.ppf(1-alpha)*stdev-rend)*np.sqrt(21)
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
##    obiettivo=sharpe(pop)
##    obiettivo=sortino_ratio(pop)
    # obiettivo_1=new_objective_function(pop,R,min_quant,max_quant,cardin_min,cardin_max)
    obiettivo=new_objective_function_2(pop,R,min_quant,max_quant,cardin_min,cardin_max)
    # obiettivo=-obiettivo
    # obiettivo=pd.Series(obiettivo.T) 
    # for index,elemnt in enumerate(obiettivo):
    #     # ##print('element is ',elemnt)
    #     obiettivo[index]=elemnt[0]
    #     # ##print('element is ',obiettivo[index])
    # ##print('objective function Paper 1 is  >>>>>>>>>',-obiettivo_1)
    ##print('objective function Paper 2 is >>>>>>>>>',-obiettivo)

    # ##print(type(obiettivo),obiettivo.shape)

    # obiettivo_omega=omega_ratio(pop)
    
    # ##print('omga objective function is ',obiettivo_omega)
    # ##print(type(obiettivo_omega),obiettivo_omega.shape)
    
    # raise ValueError
##    obiettivo=risk_parity(pop)
##    obiettivo=nuova_rm(pop)
##    obiettivo=twosided(pop)
##    obiettivo=mean_variance(pop)
##    obiettivo=mean_semivariance(pop)
##    obiettivo=mean_mad(pop)
##    obiettivo=variance_with_skewness(pop)
##    obiettivo=minimax(pop)
##    obiettivo=value_at_risk(pop)
##    obiettivo=expected_shortfall(pop)
    return obiettivo



def elitist_selection(pop, fitness, num_parents):  ##elitist selection (tasso=5%)
    ##nel loop si individua la soluzione con fitness massima
    ##il primo parent viene selezionata dalla pop con fit max
    ##la fitness viene posta molto negativa per non essere riselezionata
    parents=np.zeros((num_parents, pop.shape[1])) #inizializz dim
    for i in range(num_parents):
        posizione_max_fitness = np.where(fitness == np.max(fitness)) ##posizione idx con fitness massima
        posizione_max_fitness = posizione_max_fitness[0][0]
        parents[i,:] = pop[posizione_max_fitness, :]
        fitness[posizione_max_fitness] = -10000
    return parents

def selection(pop, fitness, num_parents):  ##elitist selection (tasso=5%)
    ##nel loop si individua la soluzione con fitness massima
    ##il primo parent viene selezionata dalla pop con fit max
    ##la fitness viene posta molto negativa per non essere riselezionata
    parents=np.zeros((num_parents, pop.shape[1])) #inizializz dim
    #for i in range(num_parents):
        #posizione_max_fitness = np.where(fitness == np.max(fitness)) ##posizione idx con fitness massima
        #posizione_max_fitness = posizione_max_fitness[0][0]
        #parents[i,:] = pop[posizione_max_fitness, :]
    parents = pop
        #fitness[posizione_max_fitness] = -10000
    return parents

#tournament selection
def tournament_selection(pop,fitness, num_parents):
    parents=np.zeros((num_parents, pop.shape[1])) #inizializz dim
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



def crossover(parents, offspring_size): #Goldberg (1975)
    offspring=np.zeros(offspring_size)
    crossover_point=int(offspring_size[1]/2) ##len colonne/2 (k_point crossover con k=1, crossover deterministico a meta' cromosoma)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        offspring[i, :crossover_point] = parents[parent1_pos, :crossover_point] ##meta' dei geni dal primo genitore
        offspring[i, crossover_point:] = parents[parent2_pos, crossover_point:] ##meta' dei geni dal secondo genitore
        ##completato il primo loop, viene generato il primo figlio
    return offspring


def two_point_crossover(parents,offspring_size): #Goldberg (1975), Muehlenberger (1993)
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

def three_point_crossover(parents,offspring_size): #Muehlenberger (1993)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        crossover_point_1=int((offspring_size[1]-1)/3)
        crossover_point_2=int((2*offspring_size[1]-1)/3)
        crossover_point_3=int((3*offspring_size[1]-1)/3)
        #crossover_point_1=np.random.randint(1,geni-1)
        #crossover_point_2=np.random.randint(crossover_point_1,geni-1)
        #crossover_point_3=np.random.randint(crossover_point_2,geni-1)
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

def crossover_uniforme(parents, offspring_size): #Spears, De Jong (1991)
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            r=np.random.uniform(0,1)
            if r>0.5:
                offspring[i,j] = parents[parent1_pos,j] 
            else:
                offspring[i,j] = parents[parent2_pos,j]
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def arnone_crossover(parents, offspring_size): #Arnone Loraschi Tettamanzi (1993)
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            r=np.random.uniform(0,1)
            if r>0.5:
                offspring[i,j] =min(parents[parent1_pos,j],parents[parent1_pos,j]*((np.sum(parents[parent1_pos])+np.sum(parents[parent2_pos]))/(2*np.sum(parents[parent1_pos]))))
            else:
                offspring[i,j] =min(parents[parent2_pos,j],parents[parent2_pos,j]*((np.sum(parents[parent1_pos])+np.sum(parents[parent2_pos]))/(2*np.sum(parents[parent2_pos]))))
        ##completato il primo loop, viene generato il primo figlio
    return offspring


def crossover_uniforme_globale(parents, offspring_size): #Dan Simon (2013)
    offspring=np.zeros(offspring_size)
    #la probabilità di scelta non è più al 50% tra un genitore e l'altro, ma 1/N:
    #si sceglie alla i-esima posizione uno dei N-esimi geni di tutta la popolazione (di genitori)
    for i in range(offspring_size[0]): ##loop per riga
        for j in range(geni):
            offspring[i,j]=random.choice(parents[:,j])
    return offspring

def flat_crossover(parents, offspring_size): #Herrera (1998)
    ##i geni del figlio derivano da j=geni estrazioni random (unif.) comprese tra i valori più piccoli
    ##dei genitori ed i valori massimi dei medesimi, riferite allo stesso gene j-esimo dei genitori
    #(es per j=0, si prendono i geni più piccoli e grandi dei genitori, a seconda di dove si collocano)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =np.random.uniform(min(parents[parent1_pos,j],parents[parent2_pos,j]),max(parents[parent1_pos,j],parents[parent2_pos,j]))
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def blend_crossover(parents, offspring_size): #Houst (1995) & Herrera (1998)
    #E' una modifica del flat crossover, con alpha=0 e' equivalente. Il parametro user-defined alpha rappresenta
    #un mix tra exploration ed exploitation. Herrera (1998) propone alpha=0.5
    alpha=0.5
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                xmin=min(parents[parent1_pos,j],parents[parent2_pos,j])
                xmax=max(parents[parent1_pos,j],parents[parent2_pos,j])
                deltax=xmax-xmin
                offspring[i,j]=np.abs(np.random.uniform(xmin-(alpha*deltax),xmax+(alpha*deltax)))
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def average_crossover(parents, offspring_size): #Nomura (1997)
    #si prende -per ogni gene del figlio- la media dei j-esimi geni dei rispettivi genitori
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =(parents[parent1_pos,j]+parents[parent2_pos,j])/2
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def multi_parent_average_crossover(parents, offspring_size): #Nomura (1997)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0] 
        for j in range(geni):
                offspring[i,j] =(parents[parent1_pos,j]+parents[parent2_pos,j]+parents[parent3_pos,j])/3
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def gene_pool_crossover(parents,offspring_size): #Muhlenbein (1993)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0]
        gene_pool=np.hstack((parents[parent1_pos],parents[parent2_pos],parents[parent3_pos]))
        for j in range(geni):
           estrazione=np.random.choice(gene_pool) 
           offspring[i,j]=estrazione
           gene_pool=np.delete(gene_pool,np.where(gene_pool==estrazione)[0][0])
    return offspring

def heuristic_crossover(parents, offspring_size): #Wright (1990)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            w=np.random.uniform(0,1)
            if parents[parent1_pos,j]>parents[parent2_pos,j]:
                offspring[i,j] =parents[parent2_pos,j]+w*(parents[parent1_pos,j]-parents[parent2_pos,j])
            else:
                offspring[i,j] =parents[parent1_pos,j]+w*(parents[parent2_pos,j]-parents[parent1_pos,j])
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def arithmetic_crossover(parents, offspring_size): #Michalewicz (1996)
    #media pesata dei geni secondo parametri user-defined
    beta=0.7
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j]
            else:
                offspring[i,j]=beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j]
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def linear_crossover(parents, offspring_size): #Wright (1990)
    #selezione di un offspring all'interno di una matrice 3xgeni secondo fitness
    #i tre offspring sono generati secondo tre diff. relazioni lineari
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
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
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def simulated_binary_crossover(parents,offspring_size): #Deb and Agrawal (1995)
    #idea di fondo è quella di estrarre i figli da una distribuzione empirica
    #secondo un parametro mu che e' un elastico tra exploration ed exploitation
    #si avranno figli più o meno o simili ai genitori a seconda di beta (e mu)
    #con beta=1 -->c.d 'stationary crossover'
    mu=0.05
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
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
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def shuffle_crossover(parents,offspring_size): ##Eshelman (1989)
    offspring=np.zeros(offspring_size)
    crossover_point=np.random.randint(1,geni-1)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        parents[parent1_pos]=np.random.permutation(parents[parent1_pos])
        parents[parent2_pos]=np.random.permutation(parents[parent2_pos])
        offspring[i, :crossover_point] = parents[parent1_pos, :crossover_point] ##meta' dei geni dal primo genitore
        offspring[i, crossover_point:] = parents[parent2_pos, crossover_point:] ##meta' dei geni dal secondo genitore
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def ring_crossover(parents,offspring_size): #Yilmaz Kaya, Murat Uyar, Ramazan Tekin (2011)
    #il metodo seleziona una lista con tre passi:
    #si fondono i parents in un'unica lista
    #si sceg2lie un punto di taglio e si procede in senso orario/antiorario
    #selezionando n elementi
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
                offspring_aux.append(aux[(start+j)%aux.shape[0]]) #clockwise
        else:
            for j in range(geni):
                offspring_aux.append(aux[(start-j)%aux.shape[0]]) #anticlockwise
        offspring[i]=offspring_aux
    return offspring

def intermediate_crossover(parents, offspring_size): #Mühlenbein, H. and Schlierkamp-Voosen, D. (1993)
    #media pesata dei geni secondo parametri user-defined
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        beta=np.random.uniform(-0.25,1.25)
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=np.abs(beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j])
            else:
                offspring[i,j]=np.abs(beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j])
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def line_crossover(parents, offspring_size): #Mühlenbein, H. and Schlierkamp-Voosen, D. (1993)
    #media pesata dei geni secondo parametri user-defined
    offspring=np.zeros(offspring_size)
    beta=np.random.uniform(-0.25,1.25)
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0] 
        for j in range(geni):
            if j%2==0:
                offspring[i,j]=np.abs(beta*parents[parent1_pos,j]+(1-beta)*parents[parent2_pos,j])
            else:
                offspring[i,j]=np.abs(beta*parents[parent2_pos,j]+(1-beta)*parents[parent1_pos,j])
         ##completato il primo loop, viene generato il primo figlio
    return offspring

def queen_bee_crossover(parents,offspring_size):  ##Karc (2004)
    ##queen bee crossover con doppio taglio
    ##si sele2ziona il vettore 'regina' (con maggiore fitness)
    ##si effettuano operazioni di crossover con gli altri parents tenendo fermo il vettore 'regina'
    offspring=np.zeros(offspring_size)
    queen_bee=np.where(calcola_pop_fitness(parents)==max((calcola_pop_fitness(parents))))[0][0]
    parent1_pos=queen_bee
    for i in range(offspring_size[0]):
        #parent1_pos=i%parents.shape[0]
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

def laplace_crossover(parents,offspring_size): #Deep and Thakur (2007a)
    #crossover è basato su insieme di estrazioni random dalla distribuzione
    #di Laplace. Come per SBX-alpha, idea di fondo è usare paramatri (a&b)
    #per muoversi tra exploration ed exploitation
    offspring=np.zeros(offspring_size)
    a=0
    b=5.0
    for i in range(offspring_size[0]): ##loop per riga
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

def parent_centric_crossover(parents, offspring_size): #Garcia Martinez et al (2008)
    #versione modificata del blend crossover (BLX-alpha), Deb et al.
    #notano che la parametrizzazione determina come sempre mix tra
    #exploitation ed exploration, producendo soluzioni piu' o meno
    #vicine ai genitori
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
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def direction_based_crossover(parents, offspring_size): #Arumugam et al (2005)
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]): ##loop per riga
        r=np.random.uniform(0,1)
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        vstack=np.vstack((parents[parent1_pos],parents[parent2_pos]))
        #fitness=np.where(calcola_pop_fitness(vstack,0.5)==max((calcola_pop_fitness(vstack,0.5))))[0][0]
        if calcola_pop_fitness(vstack)[0]>=calcola_pop_fitness(vstack)[1]:
            offspring[i]=r*(np.abs(parents[parent1_pos]-parents[parent2_pos]))+parents[parent2_pos]
        else:
            offspring[i]=r*(np.abs(parents[parent2_pos]-parents[parent1_pos]))+parents[parent1_pos]
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def geometrical_crossover(parents, offspring_size): #Michalewicz et al.(1996)
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            offspring[i,j] =np.sqrt(parents[parent1_pos,j]*parents[parent2_pos,j])
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def sphere_crossover(parents, offspring_size): #Michalewicz et al.(1996)
    alpha=0.5
    offspring=np.zeros(offspring_size)
    ###al 50% (p) il peso viene ereditato da un genitore, al 50% dall'altro (1-p), si completa un loop e si passa al figlio successivo
    for i in range(offspring_size[0]): ##loop per riga
        parent1_pos = i%parents.shape[0] ##resto genitore 1 (0, poi 1,2,3,...)
        parent2_pos = (i+1)%parents.shape[0] ##resto genitore 2 (all'ultimo step riparte da zero) 1-2-3-...-0
        for j in range(geni):
            offspring[i,j] =np.sqrt(alpha*parents[parent1_pos,j]+(1-alpha)*parents[parent2_pos,j])
        ##completato il primo loop, viene generato il primo figlio
    return offspring

def simplex_crossover(parents,offspring_size):
    offspring=np.zeros(offspring_size)
    for i in range(offspring_size[0]):
        parent1_pos=i%parents.shape[0]
        parent2_pos=(i+1)%parents.shape[0]
        parent3_pos=(i+2)%parents.shape[0]
        vstack=np.vstack((parents[parent1_pos],parents[parent2_pos],parents[parent3_pos]))
        
        # raise ValueError
        pos_peggiore_fitness=np.where(calcola_pop_fitness(vstack)==min(calcola_pop_fitness(vstack)))[0][0]
        ##print('vstack is ',vstack,pos_peggiore_fitness)
        pos_migliore_fitness=np.where(calcola_pop_fitness(vstack)==max(calcola_pop_fitness(vstack)))[0][0]
        best_parents=np.delete(vstack,(pos_peggiore_fitness),axis=0)
        centroid=np.sum(best_parents,axis=0)/(len(vstack)-1)
        offspring[i]=centroid+(np.abs(centroid-vstack[pos_peggiore_fitness]))
    return offspring

def fuzzy_crossover(parents,offspring_size): #Voigt 1995
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

def unimodal_crossover(parents,offspring_size): #Ono 1997
    offspring=np.zeros(offspring_size)
    std1=0.25
    std2=0.05
    for i in range(offspring_size[0]):
        parent1_pos = i%parents.shape[0]
        parent2_pos = (i+1)%parents.shape[0]
        parent3_pos = (i+2)%parents.shape[0]
        #x_p=0.5*(parents[parent1_pos]+parents[parent2_pos])
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
        #D=(1-(np.dot(parents[parent3_pos]-parents[parent1_pos].T,parents[parent2_pos]-parents[parent1_pos])/np.dot(np.abs(parents[parent3_pos]-parents[parent1_pos]),np.abs(parents[parent2_pos]-parents[parent1_pos])))**2)**0.5
        #D=np.dot(np.abs(parents[parent3_pos]-parents[parent1_pos]),D)
        #offspring[i,:]=np.abs(x_p+np.random.normal(0,0.25)*d+((d/np.abs(d))*D*np.random.normal(0,0.1)))
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
            if estraz_outer<(prob_mutation): ##probab pari a lunghezza cromosoma
                estraz_inner=random.uniform(0,1)
                if estraz_inner>0.5:
                    offspring_crossover[i,j]=offspring_crossover[i,j]*1.1
                else:
                    offspring_crossover[i,j]=offspring_crossover[i,j]*0.9
    return offspring_crossover

def gaussian_mutation(offspring_crossover,prob_mutation): #gaussian mutation centrata rispetto all'offspring corrente
    for i in range(offspring_crossover.shape[0]):         #search domain (0,1)  
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
                offspring_crossover[i,j]=np.random.uniform(0,1) #uniform mutation centrata nel search domain (0,1)           
    return offspring_crossover





def lista_crossover(parents,offspring_size):
    c1=crossover(parents,offspring_size)
    c2=crossover_uniforme(parents,offspring_size)
    c3=crossover_uniforme_globale(parents,offspring_size)
    c4=flat_crossover(parents,offspring_size)
    c5=blend_crossover(parents,offspring_size)
    c6=average_crossover(parents,offspring_size)
    c7=simulated_binary_crossover(parents,offspring_size)
    #c8=shuffle_crossover(parents,offspring_size)
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
    #return c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c25

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

# listone=[crossover,crossover_uniforme,heuristic_crossover,
#               laplace_crossover,queen_bee_crossover,two_point_crossover,
#                 arithmetic_crossover,geometrical_crossover,
#                 simulated_binary_crossover,average_crossover,
#                 blend_crossover,flat_crossover,
#                 crossover_uniforme_globale,three_point_crossover,
#                 linear_crossover,direction_based_crossover,unimodal_crossover,
#               fuzzy_crossover,simplex_crossover,parent_centric_normal_crossover]

listone=[crossover,crossover_uniforme,heuristic_crossover,
              laplace_crossover,two_point_crossover,
                arithmetic_crossover,geometrical_crossover,
                simulated_binary_crossover,average_crossover,
                blend_crossover,flat_crossover,
                crossover_uniforme_globale,three_point_crossover,
                linear_crossover,direction_based_crossover,unimodal_crossover,
              fuzzy_crossover,parent_centric_normal_crossover]

geni=len(data.columns)  ##geni appartenenti all'i-esima soluz (cromosoma) 2*cromosomi (es per funzione in due variabili)
# ##print(geni)
# raise ValueError
cromosomi=20 #numero soluzioni generate
num_parents=5 ##numero di genitori scelti per elitist selection
simulazioni=1
num_generazioni=1000
#dimensione popolazione
pop_size=(cromosomi,geni)

#####
#offspring_size=(pop_size[0]-parents.shape[0]


######


contatore_fitness_max=[]
contatore_fitness_media=[]
contatore_entropy=[]


#def aggregated_criteria_computation():
    #calcolo della variazione fitness media e determinazione di fwin che registra impatto applicazione criteri qualita'/diversita'
#    delta_fitness_media=df_contatore_fitness_media.diff().dropna()
#    delta_entropy=df_contatore_fitness_media.diff().dropna()
#    aux1=delta_fitness_media.mean()
#    aux2=delta_entropy.mean()
#    Fwin1=pd.DataFrame(aux1)
#    Fwin2=pd.DataFrame(aux2)
#    return Fwin1, Fwin2



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
#ph=PageHinkley(min_instances=400, delta=0.00015, threshold=0.0001, alpha=0.9999)

def aggregated_criteria_computation(fitness_modulo1,entropy_modulo1):
    #calcolo della variazione fitness media e determinazione di fwin che registra impatto applicazione criteri qualita'/diversita'
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
            Fwin1=aggregated_criteria_computation(fitness_modulo1,entropy_modulo1)[0][s] #input procedura
            Fwin2=aggregated_criteria_computation(fitness_modulo1,entropy_modulo1)[1][s] #input procedura
            x=np.linspace(0,np.max(1.5*Fwin1),100)
            m=np.tan(theta) #slope
            y=m*x
            #inizializzazione
            #dp=np.zeros(len(Fwin1)) 
            #dpp=np.zeros(len(Fwin1))
            #reward=np.zeros(len(Fwin1))
            ##calcolo distanze
            #for i in range(len(Fwin1)):
            dp=abs((Fwin2-(m*Fwin1)))/(np.sqrt(1+m**2)) #retta in forma esplicita, distanza perpendicolare
            dpp=np.sqrt((Fwin1-origine[0])**2+(Fwin2-origine[1])**2) #distanza tra due punti
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



def operator_selection(credit_function): #Probability Matching (PM)
    choose_op='pm'
    if choose_op=='pm':
        p_min=0.01
        K=num_crossover
        idx=np.random.randint(0,19)
        credito=credit_function
        if i>=2*window:
            #credito=credit_assignment()
            wheel_selection=p_min+(1-K*p_min)*(credito[i-2*window,:]/(np.sum(credito[i-2*window,:])))
            ##print(wheel_selection)
            wheel_selection=np.cumsum(wheel_selection)
            u=np.random.uniform(0,1)
            for c in range(len(wheel_selection)):
                if u<wheel_selection[c]:
                    idx=c
                    ###print(idx)
                    break

                
    elif choose_op=='mab':
        idx=i
        credito=credit_function
        C=0.00001
        aux=np.sum(matrice_memoria_operatori,axis=1)
        ##print(aux)
        if i>=2*window:
            #print(credito[i-2*window,:])
            #credito=credit_assignment()
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
        #aux=np.sum(matrice_memoria_operatori,axis=1)
        ###print(aux)
        if i>=2*window and i<num_generazioni:
            best_ip=np.where(credito[i-2*window,:]==max(credito[i-2*window,:]))
            best_ip=best_ip[0][0]
            best_ip_succ=np.where(credito[i-2*window+1,:]==max(credito[i-2*window+1,:]))
            best_ip_succ=best_ip_succ[0][0]
            ##print("\n\n\n best ip is :",best_ip)
            ##print(" best_ip_succ is :",best_ip_succ)
            #credito=credit_assignment()
            for j in range(num_crossover):
                ap_selection[i-2*window+1,j]=ap_selection[i-2*window,j]+beta*(p_min-ap_selection[i-2*window,j])
            ap_selection[i-2*window+1,best_ip_succ]=ap_selection[i-2*window,best_ip]+beta*(p_max-ap_selection[i-2*window,best_ip])
            ##print(ap_selection[i-2*window,:])
            #wheel_selection=np.cumsum(ap_selection)
            #u=np.random.uniform(0,1)
            #for c in range(len(wheel_selection)):
            #    if u<wheel_selection[c]:
            #        idx=c
    return idx


##def multi_armed_bandit(credit_function): #MAB (Multi armed bandit)
##    idx=np.random.randint(0,19)
##    credito=credit_function
##    C=5
##    aux=np.sum(matrice_memoria_operatori,axis=1)
##    if i>=2*window:
##        #credito=credit_assignment()
##        mab_selection=credito[i-2*window,:]+C*np.sqrt(np.log(aux)/np.sum(aux))
##        c=np.where(mab_selection==max(mab_selection))
##        idx=c[0][0]
##    return idx
        
def gen_portifolio(cromosomi,geni):
    solutions=np.zeros((cromosomi,geni))
    for solution in range(solutions.shape[0]):
        for gene in range(geni):
            units_short=random.random.uniform(0,1)
    pass


# main_MYCROSS()
# raise ValueError


'''

selection='elitist'
strategy='always'

for w in range(1):

    fitness_max_esterno=[]
    fitness_media_esterno=[]
    entropy_list_esterno=[]
    idx=0
    matrice_memoria_operatori=np.zeros((num_crossover,4))
    memoria_angolo=np.zeros(num_generazioni)
    fitness_individui=np.zeros((num_generazioni,cromosomi))
    contatore=0

    for k in range(simulazioni):

        #popolazione (oggetto di selezione->crossover(per generare 95 figli a partire da 5 genitori)->mutazione
        nuova_pop=np.random.uniform(low=-1, high=1, size=pop_size)
        for solution in range(nuova_pop.shape[0]):
            nuova_pop[solution,nuova_pop.shape[1]-1]=abs(nuova_pop[solution,nuova_pop.shape[1]-1])
        # nuova_pop=np.random.uniform(low=-1, high=1, size=(20,32))
        ##print(nuova_pop.shape)
        # R=rendimento_medio_set_1.mean()
        # #retern constraint

        # min_quant=-1
        # max_quant=1
        # cardin_min=nuova_pop.shape[1]//2
        # cardin_max=nuova_pop.shape[1]
        
        ##print('R is :',R)
        # ##print('>>>>>>>>>>>>>>>>',nuova_pop,'>>>>>>>>>>>>>')
        # ##print(type(nuova_pop),'>>>>>>>>>>>>>')
        # raise ValueError
        fitness_max=[]
        fitness_media=[]
        entropy_list=[]

        for i in range(10):
            ##print('\n\n\n generation >>>>>>>>>>> ',i)
            time.sleep(5)
            fitness=calcola_pop_fitness(nuova_pop)
            fitness_individui[i,:]=-fitness
            entropy=calcola_entropy(nuova_pop)
            average_fitness=np.sum(fitness)/cromosomi
            if selection=='elitist':
                parents=elitist_selection(nuova_pop,fitness,num_parents)
                #idx=15
                offspring_crossover=listone[idx](parents,offspring_size=(pop_size[0]-parents.shape[0], geni))
                ################################adaptive operator selection
                #soluzioni_crossover_15=listone[0:num_crossover](parents,offspring_size=(pop_size[0]-parents.shape[0], geni))
                soluzioni_crossover_15=np.zeros((num_crossover,cromosomi,data.shape[1]))
                ##print(' soluzioni_crossover',soluzioni_crossover_15.shape)
                # raise ValueError
                for opti in range(num_crossover):
                    soluzioni_crossover_15[opti,:,:]=listone[opti](parents,offspring_size=(pop_size[0], geni))
            ################################adaptive operator selection
            #soluzioni_crossover_15=lista_crossover2(parents,offspring_size=(pop_size[0]-parents.shape[0], geni))[0:num_crossover]
            #average_fitness_15=np.zeros(num_crossover)
            #entropy_15=np.zeros(num_crossover)
            #for u in range(num_crossover):
            #    average_fitness_15[u]=np.sum(calcola_pop_fitness(soluzioni_crossover_15[u]))/cromosomi
            #    entropy_15[u]=calcola_entropy(soluzioni_crossover_15[u])
            average_fitness_15=np.array([np.sum(calcola_pop_fitness(soluzioni_crossover_15[u]))/cromosomi for u in range(num_crossover)])
            entropy_15=np.array([calcola_entropy(soluzioni_crossover_15[u]) for u in range(num_crossover)])
            fitness_modulo1[i,:]=average_fitness_15
            entropy_modulo1[i,:]=entropy_15
            if strategy=='reactive':
                angolo=reactive_moving_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(reactive_moving_strategy()))
            elif strategy=='always':
                angolo=always_moving_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(always_moving_strategy()))
            elif strategy=='decreasing':
                angolo=decreasing_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(decreasing_strategy()))
            elif strategy=='increasing':
                angolo=increasing_strategy()
                memoria_angolo[i]=angolo
                idx=operator_selection(credit_assignment(increasing_strategy()))
            ##print('idx is ',idx)
            # raise ValueError
            if idx==18:
                idx=17
            if i>0:
                if contatore==0:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
                if memoria_angolo[i]!=memoria_angolo[i-1]:
                    contatore=contatore+1
                if contatore==1:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
                if contatore==2:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
                if contatore==3:
                    matrice_memoria_operatori[idx,contatore]=matrice_memoria_operatori[idx,contatore]+1
             ######
##            if angolo==0:
##                matrice_memoria_operatori[idx,0]=matrice_memoria_operatori[idx,0]+1
##            elif angolo==np.pi/6:
##                matrice_memoria_operatori[idx,1]=matrice_memoria_operatori[idx,1]+1
##            elif angolo==np.pi/3:
##                matrice_memoria_operatori[idx,2]=matrice_memoria_operatori[idx,2]+1
##            elif angolo==np.pi/2:
##                matrice_memoria_operatori[idx,3]=matrice_memoria_operatori[idx,3]+1
            ################################
            #offspring_crossover=lista_crossover2(parents,offspring_size=(pop_size[0], geni))[w]
            #offspring_crossover=arnone_crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], geni))
            #offspring_crossover=lista_crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], geni))[w]
            offspring_mutation=mutation(offspring_crossover,0.25)
            if selection=='elitist':
                nuova_pop[:parents.shape[0], :] = parents
                nuova_pop[parents.shape[0]:, :] = offspring_mutation
            #nuova_pop=offspring_crossover
            #nuova_pop[0][nuova_pop[0].argsort()[:5]]=0
            for j in range(len(nuova_pop)):
                #nuova_pop[j][nuova_pop[j].argsort()[:5]]=0
                nuova_pop[j]=nuova_pop[j]/np.sum(nuova_pop[j])
            pos_fitness_migliore=np.where(fitness == np.max(fitness))
            pos_fitness_migliore=pos_fitness_migliore[0][0]
            #fitness_max.append(fitness[pos_fitness_migliore])
            fitness_media.append(average_fitness)
            entropy_list.append(entropy)
        #fitness_max_esterno.append(fitness_max)
        fitness_media_esterno.append(fitness_media)
        entropy_list_esterno.append(entropy_list)
        
        
    #fitness=calcola_pop_fitness(nuova_pop,0.5) ##popolazione finale-->se ne calcola la fitness 'definitiva'
    #pos_fitness_migliore=np.where(fitness == np.max(fitness))
    #pos_fitness_migliore=pos_fitness_migliore[0][0]
    ###print("Migliore soluzione ammissibile (feasible) : ", nuova_pop[pos_fitness_migliore, :])
    ###print("Fitness : ", fitness[pos_fitness_migliore])


    
    #df_fitness_max=pd.DataFrame(np.transpose(fitness_max_esterno))
    #media_simulazioni_fitness_max=np.mean(df_fitness_max,axis=1)
    df_fitness_media=pd.DataFrame(np.transpose(fitness_media_esterno))
    media_simulazioni_fitness_media=np.mean(df_fitness_media,axis=1)
    df_entropy=pd.DataFrame(np.transpose(entropy_list_esterno))
    media_simulazioni_entropy=np.mean(df_entropy,axis=1)
    #contatore_fitness_max.append(media_simulazioni_fitness_max)
    contatore_fitness_media.append(media_simulazioni_fitness_media)
    contatore_entropy.append(media_simulazioni_entropy)
    #fig,ax = plt.subplots()
    #ax.plot(np.mean(df_fitness_media.iloc[1:,:],axis=1),c='b')
    #ax.set_ylabel('Fitness media')
    #ax.legend(['Fitness media'])
    #ax2=ax.twinx()
    #ax2.plot(np.mean(df_entropy.iloc[1:,:],axis=1),c='r')
    #ax2.set_ylabel('Entropy')
    #ax2.legend(['Entropy'])
    #plt.show()

    
    ##print(media_simulazioni_fitness_media)

    end=time.time()
    ##print(end-start)



#df_contatore_fitness_max=pd.DataFrame(np.transpose(contatore_fitness_max))
df_contatore_fitness_media=pd.DataFrame(np.transpose(contatore_fitness_media))
df_contatore_entropy=pd.DataFrame(np.transpose(contatore_entropy))
#fig,(ax1,ax2,ax3)=plt.subplots(1,3)
#ax1.plot(df_contatore_fitness_max.iloc[1:,:])
#ax1.set(ylabel='Sortino Ratio')
#ax1.set(ylabel='λE(R)-(1-λ)var(R)')
#ax1.set(xlabel='Generazioni')
#ax1.set_title('Max Fitness')
#ax2.plot(df_contatore_fitness_media.iloc[1:,:])
#ax2.set(xlabel='Generazioni')
#ax2.set_title('Fitness Media')
#ax3.plot(df_contatore_entropy.iloc[1:,:])
#ax3.set_title('Entropia')
#ax3.set(xlabel='Generazioni')
#plt.legend(['1-point crossover','uniform','global uniform',
#           'flat crossover','blend crossover','average crossover','SBX-alpha','shuffle crossover',
#           'intermediate crossover','geometrical crossover','arithmetic crossover',
#            'laplace crossover','two point crossover','queen_bee_crossover','line crossover',
#            'ring crossover','parent centric crossover','heuristic crossover'])
#plt.legend(['1-point crossover','uniform crossover','global uniform crossover',
#           'flat crossover','blend crossover','average crossover','SBX-alpha',
#            'geometrical crossover','arithmetic crossover','laplace crossover',
#            '2-point crossover','queen bee crossover','heuristic crossover',
#            '3-point crossover','line crossover'])
#plt.show()
#fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15)=plt.subplots(3,5)

lista_legenda=['1-point crossover','uniform crossover','heuristic crossover',
            'laplace crossover','queen bee crossover','two point crossover','arithmetic crossover',
            'geometrical crossover','SBX-alpha','average crossover',
            'blend crossover','flat crossover','uniform global crossover',
            '3-point crossover','line crossover']



#for i in range(5):
#    fig,(ax1,ax2,ax3)=plt.subplots(1,3)
#    fig.tight_layout(pad=4.0)
    #fig.suptitle('Risk measure: λE(R)-(1-λ)var')
#    fig.suptitle('Risk measure: Variance with Skewness')
#    ax1.plot(df_contatore_fitness_media.iloc[1:,3*i],c='b')
#    ax1.set_title(lista_legenda[3*i])
#    ax1.set_ylabel('Fitness media')
#    ax1.set_xlabel('Generazioni')
#    ax1.legend(['Fitness media'],loc=1)
#    ax11=ax1.twinx()
#    ax11.plot(df_contatore_entropy.iloc[1:,3*i],c='r')
#    ax11.set_ylabel('Entropy')
#    ax11.legend(['Entropy'],loc=4)
#    ax2.plot(df_contatore_fitness_media.iloc[1:,3*i+1],c='b')
#    ax2.set_title(lista_legenda[3*i+1])
#    ax2.set_ylabel('Fitness media')
#    ax2.set_xlabel('Generazioni')
#    ax2.legend(['Fitness media'],loc=1)
#    ax22=ax2.twinx()
#    ax22.plot(df_contatore_entropy.iloc[1:,3*i+1],c='r')
#    ax22.set_ylabel('Entropy')
#    ax22.legend(['Entropy'],loc=4)
#    ax3.plot(df_contatore_fitness_media.iloc[1:,3*i+2],c='b')
#    ax3.set_title(lista_legenda[3*i+2])
#    ax3.set_ylabel('Fitness media')
#    ax3.set_xlabel('Generazioni')
#    ax3.legend(['Fitness media'],loc=1)
#    ax33=ax3.twinx()
#    ax33.plot(df_contatore_entropy.iloc[1:,3*i+2],c='r')
#    ax33.set_ylabel('Entropy')
#    ax33.legend(['Entropy'],loc=4)
    #plt.show()
#ax4.plot(df_contatore_fitness_media.iloc[1:,3],c='b')
#ax4.set_ylabel('Fitness media')
#ax4.legend(['Fitness media'])
#ax44=ax4.twinx()
#ax44.plot(df_contatore_entropy.iloc[1:,3],c='r')
#ax44.set_ylabel('Entropy')
#ax44.legend(['Entropy'])
#ax5.plot(df_contatore_fitness_media.iloc[1:,4],c='b')
#ax5.set_ylabel('Fitness media')
#ax5.legend(['Fitness media'])
#ax55=ax5.twinx()
#ax55.plot(df_contatore_entropy.iloc[1:,4],c='r')
#ax55.set_ylabel('Entropy')
#ax55.legend(['Entropy'])
#plt.show()

##########plot probabilita selezione dell'i-esimo operatore###########
credito=credit_assignment(increasing_strategy())
roulette=np.zeros((num_generazioni-2*window,num_crossover))
p_min=0.01
for i in range(num_generazioni-2*window):
    roulette[i,:]=p_min+(1-num_crossover*p_min)*(credito[i-2*window,:]/(np.sum(credito[i-2*window,:])))
    
#fig, axs=plt.subplots(3,5)
#fig.tight_layout(pad=0.50)
#fig.suptitle('Probabilità di selezione per operatore di crossover. Misura di rischio: Sharpe Ratio',y=1.00)
#for i in range(5):
#    axs[0, i].plot(roulette[:,i])
#    axs[0, i].set_xlabel('generazioni')
#    axs[1, i].plot(roulette[:,i+5])
#    axs[1, i].set_xlabel('generazioni')
#    axs[2, i].plot(roulette[:,i+10])
#    axs[2, i].set_xlabel('generazioni')


x=[i for i in range(len(quantity_l_s_scalar_values))]
plot_vars(x,rule_R_p_scalar_values,quantity_l_s_scalar_values,card_l_s_scalar_values,l_s_total_values_scalar_values,us_reg_scalar_values,local_min_func_df,glob_min_func_df,rule_r_p_df)




fig, axs=plt.subplots(4,5)
fig.tight_layout(pad=0.50)
fig.suptitle('Probabilità di selezione. Omega + ALWAYS MOVING strategy',y=1.00)
for i in range(5):
    axs[0, i].plot(roulette[:,i])
    axs[0, i].set_xlabel('generazioni')
    axs[1, i].plot(roulette[:,i+5])
    axs[1, i].set_xlabel('generazioni')
    axs[2, i].plot(roulette[:,i+10])
    axs[2, i].set_xlabel('generazioni')
    if (i+15) < len(roulette):
        axs[3, i].plot(roulette[:,i+15])
        axs[3, i].set_xlabel('generazioni')

#####################################################################

matrice_memoria_operatori=pd.DataFrame(matrice_memoria_operatori)
mmo=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
w=0.22
sns.set()
fig, axe=plt.subplots(4,1)
fig.tight_layout(pad=0.50)
axe[0].bar(mmo-2*w,matrice_memoria_operatori.iloc[:,0],width=w,align='center')
axe[0].bar(mmo-w,matrice_memoria_operatori.iloc[:,1],width=w,align='center')
axe[0].bar(mmo,matrice_memoria_operatori.iloc[:,2],width=w,align='center')
axe[0].bar(mmo+w,matrice_memoria_operatori.iloc[:,3]+w,width=w,align='center')
if strategy=='always':
    axe[0].legend(['π/2','0','π/2','0'],loc='upper left',prop={'size': 8})
elif strategy=='reactive':
    axe[0].legend(['0','π/2','0','π/2'],loc='upper left',prop={'size': 8})
elif strategy=='decreasing':
    axe[0].legend(['π/2','π/3','π/6','0'],loc='upper left',prop={'size': 8})
elif strategy=='increasing':
    axe[0].legend(['0','π/6','π/3','π/2'],loc='upper left',prop={'size': 8})
axe[0].set_xticks(mmo)
axe[0].set_xticklabels(['OPX','UX','HX','LX','QBX','TPX','AMX','GX','SBX','AVX','BLX-a','FX','GUX','TPX','LNX','DBX','UNDX','FR','SPX','PNX'],fontsize=6,rotation=90)
axe[0].set_ylabel('Operators Frequency')
axe[1].plot(media_simulazioni_entropy[1:],c='red')
axe[1].set_ylabel('Entropy')
axe[2].plot(memoria_angolo,c='r')
axe[2].set_ylabel('Angle')
for g in range(cromosomi):
	axe[3].scatter(np.linspace(1,num_generazioni,num_generazioni-1),fitness_individui[1:,g],s=5,c=fitness_individui[1:,g],cmap='RdYlBu')
axe[3].set_ylabel('Cost')


plt.show()

'''







