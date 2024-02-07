# This file is based on 4_record_video_example and was used to create the clips for the video.
# It creates the video and saves the activations of each node per frame in a json file.

from gymnasium.wrappers import RecordVideo
from classes.manim_utils import read_file
from classes.NEAT import *
import gymnasium as gym
import json
import torch 

for gen, species_id in [
    # (0, 0), 
    (348, 0),
    # (34, 0), 
    # (66, 0), 
    # (101, 2), 
    # (117, 3), 
    # (150, 1), 
    # (172, 1), 
    # (178, 1), 
    # (212, 1), 
    # (221, 0), 
    ]:

    num_tries = 1
    run = "20240104_225458"
    path = f"src\\gymnasium_videos\\{run}\\gen{gen}species{species_id}\\"

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
                    0:torch.tensor([1.0]), # bias
                    1:torch.tensor([observation[0]]),  # x coord
                    2:torch.tensor([observation[1]]),  # y coord
                    3:torch.tensor([observation[2]]),  # x vel
                    4:torch.tensor([observation[3]]),  # y vel
                    5:torch.tensor([observation[4]]),  # angle
                    6:torch.tensor([observation[5]]),  # angular vel
                    7:torch.tensor([observation[6]]),  # left leg
                    8:torch.tensor([observation[7]]),  # right leg
                }
                actions = network.forward(input)
                actions = torch.tensor(actions)
                actions = 2*actions - 1
                actions = actions.tolist()

                log[frame] = {"input": [num for tensor in input.values() for num in tensor.tolist()], "output": actions}
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