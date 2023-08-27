import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from deap import base, creator, tools, algorithms
import random
# from ga_p_myCross_working_backup_before_cost import new_objective_function_2_indiv,plot_vars_auto,risk_function
from full_code import new_objective_function_2_indiv,plot_vars_auto,risk_function

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

num_gen= 3
num_gen_str=str(num_gen)

class Individual_glob:
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



def mutate(individual, mu, sigma, indpb, negative_prob):
    for i in range(len(individual.genotype)):
        if random.random() < indpb:
            if random.random() < negative_prob:
                individual.genotype[i] += random.gauss(-mu, sigma)  # Allow negative values
            else:
                individual.genotype[i] += random.gauss(mu, sigma)  # Allow positive values
            individual.genotype[i] = min(max(individual.genotype[i], -1.0), 1.0)  # Ensure the value is between -1 and 1
    return individual,





def main_MYCROSS_works(R=0.001):
    pop_size = (20, 28)
    population = np.random.uniform(low=-1, high=1, size=pop_size)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Individual_glob, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: np.random.rand(32).reshape((1, 32)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=5)
    toolbox.register("evaluate", evaluate_population)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, mu=0.5, sigma=0.2, indpb=0.1, negative_prob=0.2)
    toolbox.register("select", tools.selNSGA2)

    n_generations = num_gen
    variable_evolution = [population.T]
    angle_evolution = []
    entropy_evolution = []
    density_evolution = []
    cost_evolution = []

    population = [Individual_glob(individual) for individual in population]

    for gen in range(n_generations):
        print(f" \n\n\n\n GENERATION  >>>>>>>>>> {gen} \n\n\n\n")
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)

        fitness_offspring = toolbox.evaluate(offspring)

        for ind, fit in zip(offspring, fitness_offspring):
            ind.fitness.values = fit.values

        population = toolbox.select(population + offspring, k=len(population))

        evaluate_population(population)

        variable_evolution.append(np.array([ind.genotype for ind in population]).T)

        angles = np.arctan2(variable_evolution[gen][1], variable_evolution[gen][0])
        epsilon = 1e-8
        entropy_vals = entropy(variable_evolution[gen] + epsilon, axis=1)
        density = np.sum(variable_evolution[gen]) / variable_evolution[gen].size

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

    print('Im HERE>>>>>>>>>>>>>>>')

    fig, axes = plt.subplots(4, 1, figsize=(8, 6))

    axes[0].violinplot(angle_evolution.T, showmedians=True)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Angle')

    axes[1].imshow(entropy_evolution[:n_generations].T, aspect='auto', cmap='hot', origin='lower')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Entropy')

    axes[2].plot(range(n_generations), density_evolution, marker='o')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Density')

    axes[3].bar(range(n_generations), cost_evolution)
    axes[3].set_xlabel('Generation')
    axes[3].set_ylabel('Cost')

    plt.tight_layout()
    plt.savefig('/Users/abnerandreymartinezzamudio/Downloads/BSP3/BSP3/scripts/version/scriptM/results/' + num_gen_str +'_Req_R_'+str(R)+ '_95_NSGA_polts.png', dpi=3000)
    plt.clf()




from full_code import R
# print('Im HERE>>>>>>>>>>>>>>>')
main_MYCROSS_works(R=R)

from full_code import rule_R_p_scalar_values,quantity_l_s_scalar_values,card_l_s_scalar_values,l_s_total_values_scalar_values,us_reg_scalar_values,local_min_func_df,glob_min_func_df,rule_r_p_df,rule_r_p_actual_values_df
x=[i for i in range(len(quantity_l_s_scalar_values))]

plot_vars_auto(x,rule_R_p_scalar_values,quantity_l_s_scalar_values,card_l_s_scalar_values,l_s_total_values_scalar_values,us_reg_scalar_values,local_min_func_df,glob_min_func_df,rule_r_p_actual_values_df,rule_r_p_df,num_gen_str,name_of_file='NSGA_II' )










