import torch 
from classes.NEAT import *
from gymnasium.wrappers import RecordVideo #174
#386
species = torch.load('runs/continuous_lunar_lander/2023-12-04 13:29:36.670506/species_94.pt')

import gymnasium as gym
import datetime
def lunar_fitness(genotype_and_env, inputs, targets):
    #error = 0
    genotype, env = genotype_and_env
    network = NeuralNetwork(genotype) 
    fitness = 0



    num_tries = 1
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




from tqdm.auto import tqdm 
env = gym.make("LunarLander-v2")
max_fitness = -np.inf
best_genotype = None

n_workers = 8 
gymnasium_env = [gym.make("LunarLander-v2",continuous=True) for _ in range(150)]
genotypes = species[0].genotypes
with Pool(n_workers) as p:
    fitnesses = p.map(partial(lunar_fitness, inputs=None, targets=None), zip(genotypes, gymnasium_env[:len(genotypes)]))

best_genotype = genotypes[np.argmax(fitnesses)]

print(np.max(fitnesses))

env =  RecordVideo(gym.make("LunarLander-v2",continuous=True, render_mode='rgb_array') , './video')       # , render_mode='human'
env.reset() 
# while True:
fitness = lunar_fitness((best_genotype, env), None, None)
env.close()
