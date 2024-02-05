import torch 
from classes.NEAT import *
species_id = 0
run = 0


fitnesses = torch.load(f'runs/continuous_lunar_lander/20240205_165719/fitness_perspecies_{run}.pt')



import gymnasium as gym
import datetime
def lunar_fitness(genotype_and_env, inputs, targets):
    #error = 0
    genotype, env = genotype_and_env
    network = NeuralNetwork(genotype) 
    fitness = 0



    num_tries = 4
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
    return fitness






best_genotype = np.argmax(fitnesses[species_id][0])
best_genotype = fitnesses[species_id][1][best_genotype]
for species, fitness in fitnesses.items():
    print(species, np.max(fitness[0]))

env = gym.make("LunarLander-v2", render_mode='human',continuous=True)       
env.reset() 
while True:
    fitness = lunar_fitness((best_genotype, env), None, None)
