# %%
import numpy as np 

# %%
# sources of knowledge
# the website of the original author talks about issues and parameters of the algorithm https://www.cs.ucf.edu/~kstanley/neat.html
# cleared up, what mutation power is supposed to be https://digitalcommons.colby.edu/cgi/viewcontent.cgi?article=1836&context=honorstheses

# %% [markdown]
# # NEAT
# 1. Classes and Functions
# - 1.1. Neural Network (Genotype)->Phenotype, input, output dim, contains mutation: 
# - 1.2. Genotype: A->B: connection gene, A:Node gene, is_disabled, weight, keep track of gene history
# - 1.3. Crossover (Genotype1, Genotype2)->Genotype12
# - 1.4. Species, represented by random member
# - 1.5. Speciation (List of Species)-> List of Species
# - 1.6. Fitness Calculation (Species)
# - 1.7.

# %%

from copy import deepcopy
import pandas as pd

class History:
    def __init__(self):
        self.last_node_id = 0
        self.last_connection_id = 0
        self.node_innovations = {}
        self.connection_innovations = {}
    
    def add_node_gene(self, start_node_id, end_node_id): # node is added between start and end
        self.last_node_id += 1
        

class Connection_Gene_History:
    def __init__(self):
        self.innovation_number = 0
        self.history = {}
    
    def get_innovation_number(self, connection):
        if connection not in self.history:
            self.innovation_number += 1
            self.history[connection] = self.innovation_number
        
        return self.history[connection]
    
    def __contains__(self, connection):
        return connection in self.history
    
    def init_new_history(self):
        # "overwrites all previous history keys"
        temp_history = {}
        for k,v in self.history.items():
            temp_history[str(k)+'o'] = v
        
        self.history = temp_history
        

class Node_Gene_History:
    def __init__(self):
        self.innovation_number = -1
        self.history = {}
        self.node_levels = {}
    
    def get_innovation_number(self, connection, src_node, dst_node):
        
        if connection not in self.history:
            self.innovation_number += 1
            self.history[connection] = self.innovation_number
            
            dst_level = self.node_levels[dst_node]
            if self.node_levels[src_node]+1 == dst_level:
                for k,v in self.node_levels.items():
                    if v >= dst_level:
                        self.node_levels[k] +=1 # increase level of all nodes with at least dst node level
            self.node_levels[self.innovation_number] = self.node_levels[src_node]+1
        
        return self.history[connection]
    
    def add_initial_node(self, node_level, node_id=None):
        if node_id is not None:
            if self.innovation_number < node_id:
                self.innovation_number = node_id
            self.history[str(self.innovation_number)] = node_id
        else:
            self.innovation_number += 1
            self.history[str(self.innovation_number)] = self.innovation_number
        
        self.node_levels[self.innovation_number] = node_level
        
        if node_id is not None:
            return node_id
        return self.innovation_number
    
    def __contains__(self, connection):
        return connection in self.history

    def init_new_history(self):
        # "overwrites all previous history keys"
        temp_history = {}
        for k,v in self.history.items():
            temp_history[str(k)+'o'] = v
        
        self.history = temp_history


class Node_Gene:
    def __init__(self, src_node, dst_node, node_gene_history:Node_Gene_History, add_initial=False, add_initial_node_level=None, initial_node_id=None):
        connection = str(src_node)+'->'+str(dst_node)
        if add_initial:
            self.innovation_number = node_gene_history.add_initial_node( add_initial_node_level, node_id=initial_node_id)
        else:
            self.innovation_number = node_gene_history.get_innovation_number( connection, src_node, dst_node)
          
        #self.src_node = src_node
        #self.dst_node = dst_node

class Connection_Gene:
    def __init__(self, in_node, out_node, weight, is_disabled, connection_gene_history:Connection_Gene_History):
        connection = str(in_node)+'->'+str(out_node)
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.is_disabled = is_disabled
        self.innovation_number = connection_gene_history.get_innovation_number(connection)
        
            
class Genotype:
    def __init__(self, node_genes, connection_genes,
                 node_gene_history:Node_Gene_History, connection_gene_history:Connection_Gene_History,
                 mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob,mutate_remove_node_prob, mutate_add_link_prob,mutate_remove_link_prob, weight_magnitude
                 ,c1, c2, c3
                 
                 ):
        
        self.node_genes = node_genes
        self.connection_genes = connection_genes
        self.node_gene_history = node_gene_history
        self.connection_gene_history = connection_gene_history
        self.mutate_weight_prob = mutate_weight_prob
        self.mutate_weight_perturb = mutate_weight_perturb
        self.mutate_weight_random = mutate_weight_random
        self.mutate_add_node_prob = mutate_add_node_prob
        self.mutate_remove_node_prob = mutate_remove_node_prob
        self.mutate_add_link_prob = mutate_add_link_prob
        self.mutate_remove_link_prob = mutate_remove_link_prob
        self.node_genes_dict = {node_gene.innovation_number:node_gene for node_gene in self.node_genes}
        self.connection_genes_dict = {connection_gene.innovation_number:connection_gene for connection_gene in self.connection_genes}
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.weight_magnitude = weight_magnitude
    
    def print_genotype(self):
        # in pd table
        node_genes = pd.DataFrame([[node_gene.innovation_number, self.node_gene_history.node_levels[node_gene.innovation_number]] for node_gene in self.node_genes], columns=['innovation_number', 'node_level'])
        connection_genes = pd.DataFrame([[connection_gene.innovation_number, connection_gene.in_node, connection_gene.out_node, connection_gene.weight, connection_gene.is_disabled] for connection_gene in self.connection_genes], columns=['innovation_number','in_node', 'out_node', 'weight', 'is_disabled'])
        
        print('Node genes:')
        print(node_genes)
        print('Connection genes:')
        print(connection_genes)
        
        
    
    def mutate(self, mutate_add_link_prob=None):
        if mutate_add_link_prob is None:
            mutate_add_link_prob = self.mutate_add_link_prob
        mutate_remove_link_prob = self.mutate_remove_link_prob
        # mutate weight
        # random boolean  mask
        mask = np.random.rand(len(self.connection_genes)) <= self.mutate_weight_prob
        
        for connection_gene in np.array(self.connection_genes)[mask]:
            if np.random.rand() < self.mutate_weight_perturb:
                connection_gene.weight += np.random.normal() * self.weight_magnitude
                
            else:
                connection_gene.weight = np.random.normal() * self.weight_magnitude
            # clip weight
            connection_gene.weight = np.float32(np.clip(connection_gene.weight, -24, 24))
        
        # mutate add node
        if np.random.rand() < self.mutate_add_node_prob:
            self.add_node()

        # mutate add link
        if np.random.rand() < mutate_add_link_prob:
            self.add_connection()
            
        # # mutate remove node
        # technically we dont need it, since removing links is enough
        if np.random.rand() < self.mutate_remove_node_prob:
            self.remove_node()
        
        # mutate remove link
        if np.random.rand() < mutate_remove_link_prob:
            self.remove_connection()
    
    def remove_node(self):
        # select a random node gene
        if len(self.node_genes) == 0:
            return
        
        src_level = 0
        dst_level = max(self.node_gene_history.node_levels.values())
        candidate_node_genes = [gene for gene in self.node_genes if self.node_gene_history.node_levels[gene.innovation_number] != src_level and self.node_gene_history.node_levels[gene.innovation_number] != dst_level]
        if len(candidate_node_genes) == 0:
            return
        node_gene = np.random.choice(candidate_node_genes)
        self.node_genes.remove(node_gene)
        del self.node_genes_dict[node_gene.innovation_number]
        # remove connection genes
        # save the in and out node of the connection gene
        # in_nodes = []
        # out_nodes = []
        for connection_gene in self.connection_genes:
            if connection_gene.in_node == node_gene.innovation_number or connection_gene.out_node == node_gene.innovation_number:
                # if connection_gene.in_node not in in_nodes:
                #     in_nodes.append(connection_gene.in_node)
                # elif connection_gene.out_node not in out_nodes:
                #     out_nodes.append(connection_gene.out_node)
                    
                self.connection_genes.remove(connection_gene)
                del self.connection_genes_dict[connection_gene.innovation_number]
        
        # enable previous connection genes
        # for connection_gene in self.connection_genes:
        #     if connection_gene.in_node in in_nodes and connection_gene.out_node in out_nodes:
        #         connection_gene.is_disabled = False
    
    def remove_connection(self):
        # select a random connection gene
        if len(self.connection_genes) == 0:
            return
        
        connection_gene = np.random.choice(self.connection_genes)
        connection_gene.is_disabled = True
    
    def add_node(self):
        # select a random connection gene
        
        non_disabled_connection_genes = [connection_gene for connection_gene in self.connection_genes if not connection_gene.is_disabled]
        if len(non_disabled_connection_genes) == 0:
            return
        
        connection_gene = np.random.choice(non_disabled_connection_genes)
        connection_gene.is_disabled = True
        
        # add node gene
        node_gene = Node_Gene(connection_gene.in_node, connection_gene.out_node, self.node_gene_history)
        self.node_genes.append(node_gene)
        self.node_genes_dict[node_gene.innovation_number] = node_gene
        
        # add connection genes, first weight is 1.0, second is the one of the original
        connection_gene1 = Connection_Gene(connection_gene.in_node, node_gene.innovation_number, 1.0, False, self.connection_gene_history)
        connection_gene2 = Connection_Gene(node_gene.innovation_number, connection_gene.out_node, connection_gene.weight, False, self.connection_gene_history)
        self.connection_genes.append(connection_gene1)
        self.connection_genes.append(connection_gene2)
        self.connection_genes_dict[connection_gene1.innovation_number] = connection_gene1
        self.connection_genes_dict[connection_gene2.innovation_number] = connection_gene2
    
    def add_connection(self):
        permuted = np.random.permutation(self.node_genes)
        node_levels = {}
        for node_gene in permuted:
            lvl = self.node_gene_history.node_levels[node_gene.innovation_number]
            if lvl not in node_levels:
                node_levels[lvl] = [node_gene]
            else:
                node_levels[lvl].append(node_gene)
        
        for src in permuted:
            level = self.node_gene_history.node_levels[src.innovation_number]
            dsts = []
            for k, v in node_levels.items():
                if k > level:
                    dsts.extend(v)
            permuted_dst = np.random.permutation(dsts)
            for dst in permuted_dst:
                connection = str(src.innovation_number)+'->'+str(dst.innovation_number)
                
                # if connection in history but not in genotype (with or without connection being from previous evolution steps)
                # add connection gene to genotype
                # if connection not in history:
                # add connection gene to genotype
                # if connection in genotype but disabled:
                # enable connection gene
                
                connection_in_history = False
                connection_name = None
                for key in self.connection_gene_history.history.keys():
                    if key.startswith(connection):
                        connection_in_history = True
                        connection_name = key
                        break
                
                if not connection_in_history:
                    # add connection gene
                    connection_gene = Connection_Gene(src.innovation_number, dst.innovation_number, np.random.normal(), False, self.connection_gene_history)
                    self.connection_genes.append(connection_gene)
                    self.connection_genes_dict[connection_gene.innovation_number] = connection_gene
                    return
                elif self.connection_gene_history.history[connection_name] not in self.connection_genes_dict:
                    # add connection gene
                    connection_gene = Connection_Gene(src.innovation_number, dst.innovation_number, np.random.normal(), False, self.connection_gene_history)
                    self.connection_genes.append(connection_gene)
                    self.connection_genes_dict[connection_gene.innovation_number] = connection_gene
                    return
                elif self.connection_genes_dict[self.connection_gene_history.history[connection_name]].is_disabled: # if connection is disabled we take it and enable
                    self.connection_genes_dict[self.connection_gene_history.history[connection_name]].is_disabled = False
                    return
                
                
                # if connection not in the network
                # for key in self.connection_gene_history.history.keys():
                #     if key.startswith(connection):
                    
                # print(self.connection_gene_history.history)
                # if self.connection_gene_history.history[connection] not in self.connection_genes_dict:
                #     # add connection gene
                #     connection_gene = Connection_Gene(src.innovation_number, dst.innovation_number, np.random.normal(), False, self.connection_gene_history)
                #     self.connection_genes.append(connection_gene)
                #     self.connection_genes_dict[connection_gene.innovation_number] = connection_gene
                #     return # only add connection if one is possible
                # elif self.connection_genes_dict[self.connection_gene_history.history[connection]].is_disabled: # if connection is disabled we take it and enable
                #     self.connection_genes_dict[self.connection_gene_history.history[connection]].is_disabled = False
                #     return 
    
    def _crossover_genes(self, fitness_self, fitness_other, genes_self, genes_other):
        more_fit = genes_self if fitness_self > fitness_other else genes_other
        less_fit = genes_self if fitness_self < fitness_other else genes_other
        
        # create new node genes
        more_fit_innovations = set([gene.innovation_number for gene in more_fit])
        less_fit_innovations = set([gene.innovation_number for gene in less_fit])
        
        overlap = np.array(list(more_fit_innovations.intersection(less_fit_innovations)))
        disjoint = np.array(list(more_fit_innovations - less_fit_innovations)) # disjoint and excess of more fit
        
        mask = np.random.choice([True, False], len(overlap))
       
        if len(disjoint) > 0:
            from_more_fit = np.concatenate([overlap[mask], disjoint])
        else:
            from_more_fit = overlap[mask]
            
        from_less_fit = overlap[~mask]
        genes1 = [node_gene for node_gene in more_fit if node_gene.innovation_number in from_more_fit]
        genes2 = [node_gene for node_gene in less_fit if node_gene.innovation_number in from_less_fit]
        

        genes1.extend(genes2)
        return [deepcopy(gene) for gene in genes1]
    
    def crossover(self, other, fitness_self, fitness_other):
        node_genes = self._crossover_genes(fitness_self, fitness_other, self.node_genes, other.node_genes)
        connection_genes = self._crossover_genes(fitness_self, fitness_other, self.connection_genes, other.connection_genes)
        return Genotype(node_genes, connection_genes, self.node_gene_history, self.connection_gene_history, self.mutate_weight_prob, self.mutate_weight_perturb, self.mutate_weight_random, self.mutate_add_node_prob,self.mutate_remove_node_prob, self.mutate_add_link_prob,self.mutate_remove_link_prob, self.weight_magnitude, self.c1, self.c2, self.c3)
    
    
    def _distance(self, genes, genes_other):
        genes = [gene.innovation_number for gene in genes]
        genes = sorted(genes)
        other_genes = [gene.innovation_number for gene in genes_other]
        other_genes = sorted(other_genes)
        
        max_innovation_number_self = genes[-1]
        max_innovation_number_other = other_genes[-1]
        
        
        matching_genes = []
        disjoint_genes_self = []
        disjoint_genes_other = []
        excess_genes = []
        
        for innovation_number in range(min(max_innovation_number_self, max_innovation_number_other)+1):
            if innovation_number in genes and innovation_number in other_genes:
                matching_genes.append(innovation_number)
            elif innovation_number in genes:
                disjoint_genes_self.append(innovation_number)
            elif innovation_number in other_genes:
                disjoint_genes_other.append(innovation_number)
                
        excess_genes =  set(genes).union(set(other_genes)) - set(matching_genes).union(set(disjoint_genes_self)).union(set(disjoint_genes_other))
        return matching_genes, disjoint_genes_self, disjoint_genes_other, excess_genes
    
        
    def distance(self, other):
        M_n, D_self_n, D_other_n, E_n = self._distance(self.node_genes, other.node_genes)
        M_c, D_self_c, D_c, E_c= self._distance(self.connection_genes, other.connection_genes)
        
        D = len(D_self_n) + len(D_other_n) + len(D_self_c) + len(D_c)
        E = len(E_n) + len(E_c)
        
        # calculate average weight difference of matching genes
        W = 0
        for innovation_number in M_c:
            W += np.abs(self.connection_genes_dict[innovation_number].weight - other.connection_genes_dict[innovation_number].weight)
        
        W = W/len(M_c)
        N = max(len(self.node_genes) + len(self.connection_genes), len(other.node_genes)+len(other.connection_genes))
       
        return self.c1*E/N + self.c2*D/N + self.c3*W
        
    
    def __str__(self):
        return str(self.node_genes) + '\n' + str(self.connection_genes)

# %%
import torch 
from typing import Dict 



def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-4.9*x))
    
    
class NeuralNetwork(torch.nn.Module):
    def __init__(self, genotype:Genotype):
        self.genotype = genotype
        self.connection_genes = genotype.connection_genes
    
        # create nn
        self.connections_per_level = {}
        self.connections = {}
        
        
        for connection_gene in [gene for gene in self.connection_genes if not gene.is_disabled]:
            self.connections[connection_gene.innovation_number] = torch.nn.Linear(1, 1, bias=False)
            # specify weight
            self.connections[connection_gene.innovation_number].weight = torch.nn.Parameter(torch.tensor([connection_gene.weight]))
            self.connections[connection_gene.innovation_number].weight.requires_grad_(False)
            
            src_node = connection_gene.in_node
            dst_node = connection_gene.out_node
            dst_node_level = self.genotype.node_gene_history.node_levels[dst_node]
            
            if dst_node_level not in self.connections_per_level:
                self.connections_per_level[dst_node_level] = {dst_node:{}}
            elif dst_node not in self.connections_per_level[dst_node_level]:
                self.connections_per_level[dst_node_level][dst_node] = {}
            
            
            self.connections_per_level[dst_node_level][dst_node][src_node] = self.connections[connection_gene.innovation_number]
        
       
        dst_level = max([self.genotype.node_gene_history.node_levels[node.innovation_number] for node in self.genotype.node_genes])
        self.dst_level_nodes = [node.innovation_number for node in self.genotype.node_genes if self.genotype.node_gene_history.node_levels[node.innovation_number] == dst_level]
        self.dst_level_nodes = sorted(self.dst_level_nodes)
        self.sorted_levels = sorted(self.connections_per_level.keys())               
    
    
    def print_nn(self):
        # format nicely
        i = 0
        for level in reversed(self.sorted_levels):
            i+=1
            for node in self.connections_per_level[level]:
                print(' '*i*3+'Node:', node)
                for src_node in self.connections_per_level[level][node]:
                    print(' '*i*6+ 'src:', src_node, 'level:',self.genotype.node_gene_history.node_levels[src_node],'weight:', self.connections_per_level[level][node][src_node].weight.data)
    
    def forward(self, x:Dict[int, float]):
        x = deepcopy(x)
        with torch.no_grad():
            node_repr = x 
            for node in self.dst_level_nodes:
                node_repr[node] = torch.tensor([0.0]) # can happen that dst nodes have no input 
                
            for level in self.sorted_levels:
                for node in self.connections_per_level[level]:
                    input = torch.tensor([0.0])
                    for src_node in self.connections_per_level[level][node]:
                        if src_node not in node_repr:
                            continue # this happens if connections in previous layers are disabled, such that the following node has no input
                        input += self.connections_per_level[level][node][src_node](node_repr[src_node])
                            
                    node_repr[node] = sigmoid(input)
                    
            sorted_output_nodes = sorted(self.dst_level_nodes)
            return [node_repr[node] for node in sorted_output_nodes]

# nn = NeuralNetwork(genotype1)
# x = {0:torch.tensor([-1.0]),1:torch.tensor([0.3])}
# nn.forward(x)

# %%
from typing import List
from datetime import datetime 
from tqdm.auto import tqdm
class Species:
    def __init__(self, representative, genotypes, distance_delta):
        # random representative
        self.representative = representative
        self.distance_delta = distance_delta
        self.genotypes = genotypes
        self.best_fitness = -np.inf
        self.last_best_fitness_generation = 0
        self.average_distance = []

    def add_to_genotype(self, genotype):
        distance = self.representative.distance(genotype)
        self.average_distance.append(distance)
        if distance < self.distance_delta:
            self.genotypes.append(genotype)
            return True 
        else:
            return False

def get_proportional_bins(proportions, n_bins):
    
    proportions = proportions.flatten()
    proportions = proportions/sum(proportions)
    
    bins = np.round(proportions * n_bins).astype(int)
    while sum(bins) != n_bins:
        if sum(bins) > n_bins:
            # remove one in random bin
            index = np.random.choice(np.arange(len(bins)))
            bins[index] -= 1
        elif sum(bins) < n_bins:
            # add one in random bin
            index = np.random.choice(np.arange(len(bins)))
            bins[index] += 1
    
    return bins
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

from functools import partial
def evolve_once(features, target, 
                fitness_function, stop_at_fitness:float, 
                species:List[Species],fitness_survival_rate, interspecies_mate_rate, distance_delta,
                largest_species_linkadd_rate,
                eliminate_species_after_n_generations, # if no improvement 
                run_folder=None,
                generation_number=None,
                elitism=False,
                n_workers=1,
                gymnasium_env=None
                ):
    
    Node_Gene_History = species[0].representative.node_gene_history
    Connection_Gene_History = species[0].representative.connection_gene_history
    # reinit
    Node_Gene_History.init_new_history()
    Connection_Gene_History.init_new_history()
    
    #top_species_fitness = []
    top_species_adjusted_fitness = []
    #species_total_adjusted_fitness = []
    
    fittest_networks = {}

    species_with_increased_fitness_last15gens = []
    stop_marker = False
    global_max_fitness = -np.inf
    global_best_genome = (None, None)
    global_min_fitness = np.inf 
    save_fitnesses = {}
    save_crossings = {}
    for i, sp in enumerate(species):
        if len(sp.genotypes) == 0:
            continue
        
        adjusted_fitnesses = []
        fitnesses = []
        genotype_size = []
        start = datetime.now()
        with Pool(n_workers) as p:
            fitnesses = p.map(partial(fitness_function, inputs=features, targets=target), zip(sp.genotypes, gymnasium_env[:len(sp.genotypes)]))

        save_fitnesses[i] = fitnesses
        
        for fitness, genotype in zip(fitnesses, sp.genotypes):
            fitness = fitness.item()
            if fitness>global_max_fitness:
                global_max_fitness = fitness
                global_best_genome = (i, genotype)
            
            if fitness<global_min_fitness:
                global_min_fitness=fitness
            
            if fitness>=stop_at_fitness:
                if i not in fittest_networks:
                    fittest_networks[i] = []
                fittest_networks[i].append((genotype, fitness))
                stop_marker = True
                
            adjusted_fitnesses.append(fitness/len(sp.genotypes))
            fitnesses.append(fitness)
            genotype_size.append(len(genotype.connection_genes))
        end = datetime.now()
        
        
        print('Fitness calculation time:', end-start)
        # don't allow reproduction if no fitness improvement for n generations
        
        max_fitness = max(adjusted_fitnesses)
            
        if max_fitness > sp.best_fitness:
            sp.best_fitness = max_fitness
            sp.last_best_fitness_generation = 0
        else: 
            sp.last_best_fitness_generation += 1
            if sp.last_best_fitness_generation > eliminate_species_after_n_generations and not len(species) == 1:
                continue
        
        species_with_increased_fitness_last15gens.append(sp) # only allow reproduction if fitness increased
        
        
        
        #species_total_adjusted_fitness.append(sum(adjusted_fitnesses))
        mask = np.argsort(adjusted_fitnesses)
        top_n_fitness_indices = mask[-int(fitness_survival_rate*len(adjusted_fitnesses)):]
        fit_individuals = [(genotype, fitness) for genotype, fitness in zip(np.array(sp.genotypes)[top_n_fitness_indices], np.array(adjusted_fitnesses)[top_n_fitness_indices])]
        # sort by fitness
        fit_individuals = sorted(fit_individuals, key=lambda x: x[1], reverse=False)
        top_species_adjusted_fitness.append(fit_individuals)
        print(generation_number, 'Species:', i, 'mean fitness:', np.mean(fitnesses), 'best fitness:', max(adjusted_fitnesses)*len(sp.genotypes), 'worst fitness', min(adjusted_fitnesses)*len(sp.genotypes),  'average_connection_genes:', np.mean(genotype_size))   
        
        if run_folder is not None: #  and generation_number%10==0:
            with open(run_folder+f'/fitness_{generation_number}.txt', 'a+') as f:
                f.write('Species: '+str(i)+' mean fitness: '+str(np.mean(fitnesses))+' best fitness: '+str(max(adjusted_fitnesses)*len(sp.genotypes))+' average_connection_genes: '+str(np.mean(genotype_size))+'\n')
    
    if run_folder is not None:# and generation_number%10==0:
        # save species as torch pt
        torch.save(species, run_folder+f'/species_{generation_number}.pt')
            
            
    if stop_marker:
        if run_folder is not None:
            torch.save(species, run_folder+f'/species_{generation_number}.pt')
            torch.save(fittest_networks, run_folder+f'/solutions_{generation_number}.pt')
        return species, True, fittest_networks
    
    total_offsprings = sum([len(sp.genotypes) for sp in species])
    
    # fitnesses to positive range:
    # and SQUARE fitness
    top_species_adjusted_fitness = [[ (genotype,((fitness-global_min_fitness)/(global_max_fitness-global_min_fitness))**2) for (genotype, fitness) in sp] for sp in top_species_adjusted_fitness ]
    species_total_adjusted_fitness = [ sum([f for (_,f) in sp]) for sp in top_species_adjusted_fitness]
    
    proportions = np.array([species_total_adjusted_fitness[i]/sum(species_total_adjusted_fitness) for i in range(len(species_total_adjusted_fitness))])

    # inner- and interspecies mating proportions
    inter_species_number_of_offsprings = get_proportional_bins(proportions, total_offsprings)
    inner_species_number_of_offsprings_probabilities = []
    for fit_individuals ,no_offsprings in zip(top_species_adjusted_fitness, inter_species_number_of_offsprings):
        fitnesses = np.array([(fitness-global_min_fitness)/(global_max_fitness-global_min_fitness) for _, fitness in fit_individuals])
        inner_species_number_of_offsprings_probabilities.append([fitness/sum(fitnesses) for fitness in fitnesses])
    
    new_genotypes = []
    
    # interspecies mating
    if np.random.rand() < interspecies_mate_rate and len(species_with_increased_fitness_last15gens) > 1:
        # pick two species without replacement
        pair = np.random.choice(np.arange(len(species_with_increased_fitness_last15gens)), 2, replace=False)
        # pick top performers from both species
        genotype1, fitness1 = top_species_adjusted_fitness[pair[0]][-1] 
        genotype2, fitness2 = top_species_adjusted_fitness[pair[1]][-1]
        
        new_genotype = genotype1.crossover(genotype2, fitness1, fitness2)
        new_genotypes.append(new_genotype)
        # remove 1 from fit species
        if fitness1 > fitness2:
            which = pair[0]
        else:
            which = pair[1]
        
        inter_species_number_of_offsprings[which] -= 1 # remove one from fit species
    
    
  
    largest_species = [False for _ in range(len(species_with_increased_fitness_last15gens))]
    largest_species[np.argmax([len(sp.genotypes) for sp in species_with_increased_fitness_last15gens])] = True
    # innerspecies mating
    # we implement it using probablitlies to select parents
    for species_index, (fit_individuals, no_offsprings, probabilities, is_largest_species) in enumerate(zip(top_species_adjusted_fitness, inter_species_number_of_offsprings, inner_species_number_of_offsprings_probabilities,largest_species)):
        fit_individuals = [genotype for genotype, _ in fit_individuals]
        
        if species_index == global_best_genome[0] and elitism: 
            # if the species contains the best genome, we add it to the offsprings without mutation
            no_offsprings -= 1
            new_genotypes.append(global_best_genome[1])
            
        # 25% of offsprings are without crossover
        without_crossover = int(0 * no_offsprings)
        
        for i in range(no_offsprings):
            
            
            # 25% of offsprings are without crossover
            if i<=without_crossover:
                parent1 = np.random.choice(fit_individuals, 1, replace=False, p=probabilities)[0]
                parent2 = parent1
                
            elif len(fit_individuals) == 1: # if there is a new species, there will only be one individual in it
                parent1 = fit_individuals[0]
                parent2 = fit_individuals[0]
            else: # pick two parents
                parent1, parent2 = np.random.choice(fit_individuals, 2, replace=False, p=probabilities)
            
            save_crossings[i] = (parent1, parent2)
            new_genotype = parent1.crossover(parent2, 1, 1)
            # mutate
            if is_largest_species:
                new_genotype.mutate(mutate_add_link_prob=largest_species_linkadd_rate)
            else:
                new_genotype.mutate()
            new_genotypes.append(new_genotype)
    
    # remove old genotypes, except reference genotype
    for sp in species:
        sp.genotypes = []
    
    # speciate
    new_species = species_with_increased_fitness_last15gens
    for genotype in new_genotypes:
        added = False
        for sp in new_species:
            added = sp.add_to_genotype(genotype)
            if added:                
                break
            
        if not added:
            new_species.append(Species(genotype, [genotype], distance_delta))
    
    for i, sp in enumerate(new_species):
        print(i, 'average distance', np.mean(sp.average_distance))
        sp.average_distance = []
    
    torch.save(save_fitnesses, run_folder+f'/fitness_perspecies_{generation_number}.pt')
    torch.save(save_crossings, run_folder+f'/crossings_{generation_number}.pt')
        
    return new_species, False, None

import os 
def evolve(features, target, fitness_function, stop_at_fitness:float, n_generations, species:Species, fitness_survival_rate, interspecies_mate_rate, distance_delta, largest_species_linkadd_rate, eliminate_species_after_n_generations, run_folder=None, elitism=False, n_workers=1, gymnasium_env=None):
    if run_folder is not None:
        os.makedirs(run_folder)
        
    for i in range(n_generations):
        species, found_solution, solutions = evolve_once(features, target, fitness_function, stop_at_fitness, species, fitness_survival_rate, interspecies_mate_rate, distance_delta, largest_species_linkadd_rate, eliminate_species_after_n_generations, run_folder, i, elitism, n_workers, gymnasium_env)
        if found_solution:
            print('Found solution in generation', i)
            for k,v in solutions.items():
                print('Species:', k)
                for genotype, fitness in v:
                    print('Fitness:', fitness)
                    print(genotype.print_genotype())
            return species, solutions

    return species, None

# %%
def xor_fitness(network:NeuralNetwork, inputs, targets, print_fitness=False):
    #error = 0
    fitness = 4
    for input, target in zip(inputs, targets):
        output = network.forward(input)[0]
        if output is None:
            return torch.tensor([0]) # if network has no connected nodes (all are disabled)
        fitness -= (output - target)**2
    
   
    return fitness

# %%

# if __name__ == '__main__':
#     import random 
#     random.seed(14)
#     torch.manual_seed(14)
#     np.random.seed(14)
    
#     n_networks = 150
    

#     # Fitness:
#     c1 = 1
#     c2 = 1
#     c3 = 0.4
#     distance_delta = 6


#     weight_magnitude = 2.5 # std of weight mutation
#     # Mutation
#     mutate_weight_prob = 0.8
#     mutate_weight_perturb = 0.8
#     mutate_weight_random = 1 - mutate_weight_perturb
#     mutate_add_node_prob = 0.02
#     mutate_remove_node_prob = 0.02
#     mutate_add_link_prob_large_pop = 0.08
#     mutate_add_link_prob = 0.02
#     mutate_remove_link_prob = 0.02

#     offspring_without_crossover = 0.25
#     interspecies_mate_rate = 0.001

#     fitness_survival_rate = 0.2
#     interspecies_mate_rate = 0.001


#     node_gene_history = Node_Gene_History()
#     connection_gene_history = Connection_Gene_History()

#     genotypes = []


#     for _ in range(n_networks):
#         node_genes = [
#             Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=0), 
#             Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=1),
#             Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=0, initial_node_id=2),
#             Node_Gene(None, None, node_gene_history, add_initial=True, add_initial_node_level=1, initial_node_id=3)
#         ]
        
#         connection_genes = [
#             Connection_Gene(0, 3, np.random.normal(), False, connection_gene_history), # bias
#             Connection_Gene(1, 3, np.random.normal(), False, connection_gene_history), # input 1 
#             Connection_Gene(2, 3, np.random.normal(), False, connection_gene_history), # input 2
#         ]
        
#         genotype = Genotype(
#             node_genes, connection_genes, node_gene_history, connection_gene_history, 
#             mutate_weight_prob, mutate_weight_perturb, mutate_weight_random, mutate_add_node_prob, mutate_remove_node_prob,  mutate_add_link_prob, mutate_remove_link_prob, weight_magnitude,
#             c1, c2, c3)
#         genotypes.append(genotype)

#     # %%
#     # xor
#     inputs = [
#         {0:torch.tensor([1.0]),1:torch.tensor([0.0]),2:torch.tensor([0.0])},
#         {0:torch.tensor([1.0]),1:torch.tensor([1.0]),2:torch.tensor([0.0])},
#         {0:torch.tensor([1.0]),1:torch.tensor([0.0]),2:torch.tensor([1.0])},
#         {0:torch.tensor([1.0]),1:torch.tensor([1.0]),2:torch.tensor([1.0])}
#         # xor:
#         # bias 1, 00, 01, 10, 11
#     ]
#     targets = [
#         torch.tensor([0.0]),
#         torch.tensor([1.0]),
#         torch.tensor([1.0]),
#         torch.tensor([0.0])
#     ]

#     # %%
#     len(genotypes)

#     # %%
    
    
#     initial_species = Species(np.random.choice(genotypes), genotypes, distance_delta)

#     evolved_species, solutions = evolve(
#         features=inputs, 
#         target=targets, 
#         fitness_function=xor_fitness, 
#         stop_at_fitness=3.85, 
#         n_generations=1000,
#         species=[initial_species], 
#         fitness_survival_rate=fitness_survival_rate, 
#         interspecies_mate_rate=interspecies_mate_rate, 
#         distance_delta=distance_delta,
#         largest_species_linkadd_rate=mutate_add_link_prob_large_pop,
#         eliminate_species_after_n_generations=20
        
#     )

#     print(solutions)
