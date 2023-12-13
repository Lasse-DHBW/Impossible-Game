import random
import torch
from classes.NEAT import *
import os


def get_generation_info(generation, run):
    prefix = "." if os.getcwd().split("\\")[-1] == "src" else "src"  # cwd is either impossible-game or src depending on if the file is executed via cmd terminal, manim_sideview or interactive terminal
    species_list = torch.load(f'{prefix}/runs/continuous_lunar_lander/{run}/species_{generation}.pt')
    fitness_per_species = torch.load(f'{prefix}/runs/continuous_lunar_lander/{run}/fitness_perspecies_{generation}.pt')

    return (species_list, fitness_per_species)


def decypher_genotype(genotype):
    node_genes = pd.DataFrame([[node_gene.innovation_number, genotype.node_gene_history.node_levels[node_gene.innovation_number]] for node_gene in genotype.node_genes], columns=['innovation_number', 'node_level'])
    connection_genes = pd.DataFrame([[connection_gene.innovation_number, connection_gene.in_node, connection_gene.out_node, connection_gene.weight, connection_gene.is_disabled] for connection_gene in genotype.connection_genes], columns=['innovation_number','in_node', 'out_node', 'weight', 'is_disabled'])

    vertices = node_genes['innovation_number'].to_numpy()
    partitions = [list(node_genes[node_genes['node_level'] == i]['innovation_number']) for i in range(node_genes['node_level'].max() + 1)]
    edges = [(row[1]['in_node'], row[1]['out_node']) for row in connection_genes.iterrows()]

    return (vertices, edges, partitions)



# Sanity Checks
run = '2023_12_07_19_59_20_340927'

species_list, fitness_per_species = get_generation_info(generation=0, run=run)
print("Number of species: ", len(species_list))

vertices, edges, partitions = decypher_genotype(species_list[0].representative)

print("Vertices: ", vertices)
print("Edges: ", edges)
print("Partitions: ", partitions)

