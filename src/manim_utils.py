import random
import torch
from classes.NEAT import *
import os
from manim import * 

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



