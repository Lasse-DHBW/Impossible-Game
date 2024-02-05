# This file runs a neat algorithm run for finding the best lunar lander

import random 
import torch 
import numpy as np
from classes.NEAT import *
import gymnasium as gym

random.seed(14)
torch.manual_seed(14)
np.random.seed(14)


n_networks = 150


# Specify speciation parameters:
c1 = 1
c2 = 1
c3 = 0.5
distance_delta = 4


# Specifiy Mutation weights

weight_magnitude = 1.5 # std of weight mutation
mutate_weight_prob = 0.8
mutate_weight_perturb = 0.8
mutate_weight_random = 1 - mutate_weight_perturb
mutate_add_node_prob = 0.7
mutate_remove_node_prob = 0.7
mutate_add_link_prob_large_pop = 0.5
mutate_add_link_prob = 0.5
mutate_remove_link_prob = 0.5

interspecies_mate_rate = 0.001
fitness_survival_rate = 0.2
interspecies_mate_rate = 0.001




# initialize genotypes
node_gene_history = Node_Gene_History()
connection_gene_history = Connection_Gene_History()

genotypes = []

for _ in range(n_networks):
    
    node_genes = []
    for i in range(9):
        node_genes.append(Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=i))
    
    for i in range(9,11):
        node_genes.append(Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=1, initial_node_id=i))
    
    connection_genes = []
    # initially only connect bias to output
    connection_genes.append(Connection_Gene(0, 9, np.random.normal(), False, connection_gene_history))    
    connection_genes.append(Connection_Gene(0, 10, np.random.normal(), False, connection_gene_history))    
    
    
    genotype = Genotype(
        node_genes, connection_genes, node_gene_history, connection_gene_history, 
        mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob,mutate_remove_node_prob, mutate_add_link_prob,mutate_remove_link_prob, weight_magnitude,
        c1, c2, c3)
    genotypes.append(genotype)






# Fitness function for lunar lander
def lunar_fitness(genotype_and_env, inputs, targets):
    number,genotype, env = genotype_and_env
    network = NeuralNetwork(genotype) 
    fitness = 0
    
    num_tries = 5
    for _ in range(num_tries):
        observation, info = env.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            input = {
                0:torch.tensor([1.0]),# bias
                1:torch.tensor([observation[0]]),
                2:torch.tensor([observation[1]]),
                3:torch.tensor([observation[2]]),
                4:torch.tensor([observation[3]]), 
                5:torch.tensor([observation[4]]),
                6:torch.tensor([observation[5]]),
                7:torch.tensor([observation[6]]),
                8:torch.tensor([observation[7]]), 
            }
            actions = network.forward(input)
            actions = torch.tensor(actions)
            actions = 2*actions - 1
            actions = actions.tolist()
            observation, reward, terminated, truncated, info = env.step(actions)
            fitness += reward
            
    fitness /= num_tries
    env.close()
    return fitness, number


initial_species = Species(np.random.choice(genotypes), genotypes, distance_delta, 0)


import os
import datetime 
now = datetime.datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")
folder = os.path.join('runs', 'continuous_lunar_lander', str(now))
    
evolved_species, solutions = evolve(
    features=None, 
    target=None, 
    fitness_function=lunar_fitness, 
    stop_at_fitness=1000, # this is never reached, so evolove forever
    n_generations=10000,
    species=[initial_species], 
    fitness_survival_rate=fitness_survival_rate, 
    interspecies_mate_rate=interspecies_mate_rate, 
    distance_delta=distance_delta,
    largest_species_linkadd_rate=mutate_add_link_prob_large_pop,
    eliminate_species_after_n_generations=20,
    run_folder=folder,
    n_workers=16,
    gymnasium_env=[gym.make( "LunarLander-v2",continuous=True) for _ in range(n_networks)],
    elitism=True  # keep the best network from the previous generation
    
)
