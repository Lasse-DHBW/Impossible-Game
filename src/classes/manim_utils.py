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
    # used to translate the genotype into a dataframe for further use
    node_genes = pd.DataFrame([[node_gene.innovation_number, genotype.node_gene_history.node_levels[node_gene.innovation_number]] for node_gene in genotype.node_genes], columns=['innovation_number', 'node_level'])
    connection_genes = pd.DataFrame([[connection_gene.innovation_number, connection_gene.in_node, connection_gene.out_node, connection_gene.weight, connection_gene.is_disabled] for connection_gene in genotype.connection_genes], columns=['innovation_number','in_node', 'out_node', 'weight', 'is_disabled'])

    return (node_genes, connection_genes)

class CText(Text):
    # Text with custom font
    def __init__(self, text, font="Montserrat", **kwargs):
        super().__init__(text, font=font, **kwargs)

def get_direction(angle):
    # Returns the direction vector of the given angle
    return np.array([-np.sin(angle), np.cos(angle), 0])

def calc_lag_ratio(total_duration, subanim_duration, num_subanims):
    # calculates how much lag a succession animation with n subanimations should have in order to last a certain <total_duration>
    last_subanimation_start_time = total_duration - subanim_duration
    lag_ratio = last_subanimation_start_time / (num_subanims - 1)
    return lag_ratio

def analyse_run(run, num_species=None):
    """
    Used to analyse whole runs located in the \runs folder. Creates two plots and two visualizations, 
    that contain information various informations, i.e. the fitness of each species in each generation.
    However this only really yields interesting results if all generations of a run are present in the file repository.
    We had to remove all generations that were not strictly relevant for the animation in order to keep the repo size under 1 GB.
    So using this function without creating a new run first, will not be that interesting.
    """
    data = []
    species_set = set()

    prefix = "." if os.getcwd().split("\\")[-1] == "src" else "src"
    for filename in os.listdir(f"{prefix}\\runs\\continuous_lunar_lander\\{run}"):

        if filename.startswith("fitness_perspecies"):
            fitness_perspecies = torch.load(f"{prefix}\\runs\\continuous_lunar_lander\\{run}\\{filename}")

            file_data = {'Generation': int(filename[19:-3])}
            best_species = None
            best_max_fitness = float('-inf')
            best_genotype = None

            for species, (fitnesses, genotypes) in fitness_perspecies.items():
                species_set.add(species)
                max_fitness = max(fitnesses)
                max_index = fitnesses.index(max_fitness)
                file_data[f'species_{species}_min'] = min(fitnesses)
                file_data[f'species_{species}_mean'] = sum(fitnesses) / len(fitnesses)
                file_data[f'species_{species}_max'] = max_fitness
                file_data[f'species_{species}_num_genotypes'] = len(genotypes)

                # Check if this species has the best max fitness
                if max_fitness > best_max_fitness:
                    best_max_fitness = max_fitness
                    best_species = species
                    best_genotype = genotypes[max_index]

            file_data['best_species'] = best_species
            file_data['best_genotype'] = best_genotype
            file_data['best_fitness'] = best_max_fitness

            data.append(file_data)

    df = pd.DataFrame(data).sort_values("Generation")
    df.set_index("Generation", inplace=True)

    # Visualizations
    if num_species is not None:
        species_set = list(species_set)[:num_species]

    # ========== Visualisation of max fitness per species

    # df = df[:163]  # <- second best run, tiny deviation to best
    for col in [f"species_{species}_max" for species in species_set]:
        df.loc[:,col] = df[col].clip(lower=-500, upper=300)
        # clips 1 value in species 1 and 2 each (outlier removal)

    # Assuming your DataFrame has columns named 'Column1', 'Column2', 'Column3'
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    for species in species_set:
        plt.plot(list(df[f'species_{species}_max']), label=f'species_{species}_max')

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


    # ========== Visualisation of num of genotypes per species per gen

    cols = [f"species_{species}_num_genotypes" for species in species_set]
    analysis2 = df[cols]
    analysis2 = analysis2.fillna(0)
    analysis2["sum"] = analysis2[cols].sum(axis=1)

    # Assuming your DataFrame has columns named 'Column1', 'Column2', 'Column3'
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    for species in species_set:
        plt.plot(list(analysis2[f'species_{species}_num_genotypes']), label=f'species_{species}_num_genotypes')

    xticks = np.arange(0, df.__len__(), 5)
    plt.xticks(xticks, rotation=90) 

    # Adding titles and labels
    plt.xlabel('Index')
    plt.ylabel('Values')

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()

    return df, analysis2