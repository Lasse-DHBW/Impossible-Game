from gymnasium.wrappers import RecordVideo
from manim_utils import read_file
from classes.NEAT import *
import gymnasium as gym
import json
import torch 

gen = 75
species_id = 2
num_tries = 1
run = "20231220_115303"
path = f"src\\gymnasium_videos\\gen{gen}species{species_id}\\"

species = read_file(generation=gen, run=run, file="species")
fitnesses = read_file(generation=gen, run=run, file="fitness_perspecies")

def lunar_fitness(genotype_and_env, num_tries):
    genotype, env = genotype_and_env
    network = NeuralNetwork(genotype) 
    fitness = 0

    log = {}
    for _ in range(num_tries):
        observation, info = env.reset()
        terminated, truncated = False, False
        frame = 0
        while not terminated and not truncated:
            frame += 1
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

            input_reordered = [input[i] for i in range(8, 0, -1)]
            input_reordered.insert(0, input[0])
            log[frame] = {"input": [num for tensor in input_reordered for num in tensor.tolist()], "output": actions}
            with open(path + "log.json", "w") as f:
                json.dump(log, f)

            observation, reward, terminated, truncated, info = env.step(actions)
            fitness += reward
            
    fitness /= num_tries
    return fitness


genotypes = species[species_id].genotypes

best_genotype = genotypes[np.argmax(fitnesses[species_id][0])]
best_genotype.print_genotype()

env =  RecordVideo(gym.make("LunarLander-v2",continuous=True, render_mode='rgb_array') , path)
    
env.reset() 
fitness = lunar_fitness((best_genotype, env), num_tries=num_tries)
env.close()