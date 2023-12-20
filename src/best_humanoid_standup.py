import torch 
from classes.NEAT import *
species_id = 0
run = 1

species = torch.load(f'runs/humanoid_standup/2023-12-08 21:18:08.795747/species_{run}.pt')
fitnesses = torch.load(f'runs/humanoid_standup/2023-12-08 21:18:08.795747/fitness_perspecies_{run}.pt')



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
            }
            for i in range(1,45):
                input[i] = torch.tensor([observation[i-1]]).to(torch.float32)
                
            actions = network.forward(input)
            actions = torch.tensor(actions)
            actions = 2*actions - 1
            # normalize to -0.4, 0.4
            actions = actions * 0.4
            #actions = actions.tolist()
            observation, reward, terminated, truncated, info = env.step(actions)
            fitness += reward
            
    fitness /= num_tries
    env.close()
    return fitness




genotypes = species[species_id].genotypes

best_genotype = genotypes[np.argmax(fitnesses[species_id])]

for species in fitnesses:
    print(species, np.max(fitnesses[species]))
#best_genotype.print_genotype()
# env = gym.make("HumanoidStandup-v4", render_mode='human')       
# env.reset() 
# while True:
#     fitness = lunar_fitness((best_genotype, env), None, None)
    
from gymnasium.wrappers import RecordVideo

env = RecordVideo(gym.make("HumanoidStandup-v4", render_mode='rgb_array')   , './video')     

# while True:
fitness = lunar_fitness((best_genotype, env), None, None)
env.close()