import random
import torch
from classes.NEAT import *
import os
from manim import * 
import pandas as pd
import matplotlib.pyplot as plt



def read_file(generation, run, file):
    prefix = "." if os.getcwd().split("\\")[-1] == "src" else "src"  # cwd is either impossible-game or src depending on if the file is executed via cmd terminal, manim_sideview or interactive terminal
    return torch.load(f'{prefix}/runs/continuous_lunar_lander/{run}/{file}_{generation}.pt')

def decypher_genotype(genotype):
    node_genes = pd.DataFrame([[node_gene.innovation_number, genotype.node_gene_history.node_levels[node_gene.innovation_number]] for node_gene in genotype.node_genes], columns=['innovation_number', 'node_level'])
    connection_genes = pd.DataFrame([[connection_gene.innovation_number, connection_gene.in_node, connection_gene.out_node, connection_gene.weight, connection_gene.is_disabled] for connection_gene in genotype.connection_genes], columns=['innovation_number','in_node', 'out_node', 'weight', 'is_disabled'])

    return (node_genes, connection_genes)

class CText(Text):
    def __init__(self, text, font="Montserrat", **kwargs):
        super().__init__(text, font=font, **kwargs)

def get_direction(angle):
    return np.array([-np.cos(angle), np.sin(angle), 0])

def calc_lag_ratio(total_duration, subanim_duration, num_subanims):
    last_subanimation_start_time = total_duration - subanim_duration
    lag_ratio = last_subanimation_start_time / total_duration / (num_subanims - 1)
    return lag_ratio

def analyse_run(run):
    data = []

    prefix = "." if os.getcwd().split("\\")[-1] == "src" else "src"
    for filename in os.listdir(f"{prefix}\\runs\\continuous_lunar_lander\\{run}"):

        if filename.startswith("fitness_perspecies"):
            fitness_perspecies = torch.load(f"{prefix}\\runs\\continuous_lunar_lander\\{run}\\{filename}")

            file_data = {'Generation': int(filename[19:-3])}
            best_species = None
            best_max_fitness = float('-inf')
            best_genotype = None

            for species, (fitnesses, genotypes) in fitness_perspecies.items():
                max_fitness = max(fitnesses)
                max_index = fitnesses.index(max_fitness)
                file_data[f'species_{species}_min'] = min(fitnesses)
                file_data[f'species_{species}_mean'] = sum(fitnesses) / len(fitnesses)
                file_data[f'species_{species}_max'] = max_fitness

                # Check if this species has the best max fitness
                if max_fitness > best_max_fitness:
                    best_max_fitness = max_fitness
                    best_species = species
                    best_genotype = genotypes[max_index]

            file_data['best_species'] = best_species
            file_data['best_genotype'] = best_genotype

            data.append(file_data)

    df = pd.DataFrame(data).sort_values("Generation")


    # Visualisation of max fitness per species
    df = df[:163]  # <- second best run, tiny deviation to best

    for col in ["species_0_max", "species_1_max", "species_2_max", "species_3_max"]:
        df.loc[:,col] = df[col].clip(lower=-500, upper=300)
        # clips 1 value in species 1 and 2 each (outlier removal)

    # Assuming your DataFrame has columns named 'Column1', 'Column2', 'Column3'
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.plot(list(df['species_0_max']), label='species_0_max')
    plt.plot(list(df['species_1_max']), label='species_1_max')
    plt.plot(list(df['species_2_max']), label='species_2_max')
    plt.plot(list(df['species_3_max']), label='species_3_max')

    xticks = np.arange(0, df.__len__(), 5)
    plt.xticks(xticks, rotation=90) 
    yticks = np.arange(-500, 300, 50)
    plt.yticks(yticks) 

    # Adding titles and labels
    plt.xlabel('Index')
    plt.ylabel('Values')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()


    return df