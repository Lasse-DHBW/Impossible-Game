import os
from manim import *
from manim_utils import read_file, decypher_genotype
from manim_mobjects import Cell, Nucleus
from manim_utils import CText
import pandas as pd
import random

# Debugging
random.seed = 42

color_highlight = YELLOW_D
color_mutation = BLUE_D

# Neat
run = '2023_12_07_19_59_20_340927'

class Scene00Setup(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#222222"
        self.camera.frame.scale_to_fit_height(16)



class Scene02Title(Scene00Setup):
    def construct(self):
        super().construct()

        title = CText("NEAT", weight=BOLD).shift(UP).scale(2)
        subtitle = CText("NeuroEvolution of Augmenting Topologies").scale(0.8)

        self.play(Write(title))
        self.wait(1)
        self.play(Write(subtitle))


class Scene03Individual(Scene00Setup):
    def construct(self):
        super().construct()

        # Fetch individuum from first generation
        species_list = read_file(generation=0, run=run, file="species")
        genes = decypher_genotype(species_list[0].genotypes[0])

        """
        To understand how the NEAT algorithm actually works, we'll have to jump back to the very beginning 
        of the evolutionary process, starting with the most basic network imaginable for the task at hand. 
        """

        # Create cell
        cell_height = 4
        cell = Cell(width=cell_height/2, height=cell_height, genes=genes, num_waves=6, wobble_frequency=3)
        cell.set_opacity(0)
        cell.membrane.set_stroke(opacity=0)
        cell.nucleus.set_opacity(0)
        self.play(
            cell.membrane.animate.set_stroke(opacity=1),
            cell.nucleus.animate.set_opacity(0.2),
            run_time=3
            )

        # Zoom in
        v_margin, h_margin = 1.5, 0.5
        self.play(self.camera.frame.animate.scale_to_fit_height(cell_height + v_margin*2), run_time=1)

        # self.play(self.camera.resize_frame_shape(height=cell.height + 1), run_time=1)
        # self.play(self.camera.auto_zoom([cell], margin=0.5), run_time=1)

        """
        Its nodes represent neurons, similar to those in our brain, and the pathways are the synaptic connections. 
        Our organism's brain, if you will, comprises nine input nodes, mirroring sensory information from the 
        Lunar Lander game – the lander's position, velocity, angle, and contact sensors, plus an additional bias node, 
        always set to one.
        """

        # Highlight input nodes
        for vertice in cell.nucleus.vertices:
            if vertice in cell.nucleus.partitions[0]:
                cell.nucleus[vertice].set_opacity(1)
                self.wait(0.2)

        """
        But what about the outputs? This organism has to make decisions, after all. 
        Here, it boils down to two critical numbers: one controlling the main engine, another dictating the 
        lateral thrusters. These outputs are its actions, its means of interacting with its environment – 
        in this case, the challenging terrain of the lunar surface.
        """

        # Highlight output nodes
        for vertice in cell.nucleus.vertices:
            if vertice in cell.nucleus.partitions[1]:
                cell.nucleus[vertice].set_opacity(1)
                self.wait(0.2)

        """
        This most basic neural network there are only 2 connections, both connecting one of the outputs 
        with the bias node.  meaning that the inputs aren't influencing the actions of the lunar lander at all. 
        This is obviously a pretty poor architecture, since the gamestate is in no way contributing to the 
        actions that the lunar lander takes, meaning that information like the landers coordinates and velocity 
        are entirely disregarded.
        """

        # Highlight connections
        for i, edge in enumerate(cell.nucleus.edges):
            # self.play(cell.nucleus.edges[edge].animate.set_color(YELLOW), run_time=1)
            cell.nucleus.edges[edge].set_opacity(1)
        
        self.wait(1)

        """
        This issue will soon be adressed through the process of mutation. But before we look into that, 
        we must understand how the current architecture of the network is expressed in the NEAT framework. 
        """

        # Move to side
        new_cell_pos = LEFT*(self.camera.frame_width / 2 - cell.width / 2 - h_margin) 
        self.play(cell.animate.move_to(new_cell_pos))       

        # Add divider
        divider = DashedLine(
            self.camera.frame_height // 2 * UP + LEFT*(self.camera.frame_width / 2 - cell.width - h_margin*2), 
            self.camera.frame_height // 2 * DOWN + LEFT*(self.camera.frame_width / 2 - cell.width - h_margin*2),  
            dashed_ratio=0.3, 
            color=WHITE
            )
        
        self.play(
            Create(divider),
            cell.membrane.animate.set_opacity(0.5),
            cell.nucleus.edges[(0, 10)].animate.set_opacity(0.51),
            run_time = 1
            )

        self.wait(1)

        """
        =========== GRID - GENOTYPE / PHENOTYPE ===========
        First we look at its node gene. There is a node gene for each node of 
        the network expressing which layer it belongs to. 
        """

        grid = {"divider": divider}
        grid_font_scaling = 0.15
        grid_padding = 0.2
        
        node_genes_vis, conn_genes_vis = cell.nucleus.get_visual_genotype(font_scaling=grid_font_scaling).values()
        grid["conn_genes"] = VGroup(*conn_genes_vis)

        starting_position = UP*(self.camera.frame_height/2 - v_margin) + LEFT*(self.camera.frame_width/2 - cell.width - h_margin*3)

        node_gene_title = CText("Node Genes").scale(0.25)
        grid["node_gene_title"] = node_gene_title
        node_gene_title.move_to(starting_position + DOWN*(node_gene_title.height/2) + RIGHT*(node_gene_title.width/2))
        self.play(Write(node_gene_title), run_time=1)
        starting_position = starting_position + DOWN*(node_gene_title.height + grid_padding)
        
        input_layer = [node_gene for node_gene in node_genes_vis if node_gene[0][1].text == "InputLayer"]
        hidden_layer = [node_gene for node_gene in node_genes_vis if node_gene[0][1].text[:11] == "HiddenLayer"]
        output_layer = [node_gene for node_gene in node_genes_vis if node_gene[0][1].text == "OutputLayer"]
        
        grid.update({
            "input_layer": VGroup(*input_layer),
            "output_layer": VGroup(*output_layer)
        })

        for layer in [input_layer, hidden_layer, output_layer]:
            if layer.__len__() == 0:
                continue
            
            for i, node_gene in enumerate(layer):
                right_shift = i * (node_gene.width + grid_padding)
                node_gene.move_to(
                    starting_position + 
                    DOWN*(node_gene.height/2) + 
                    RIGHT*(node_gene.width/2) + RIGHT*right_shift
                    )
                self.play(Create(node_gene), run_time=0.1)

            starting_position = starting_position + DOWN*(layer[0].height + grid_padding)

        """
        Additionally there is one connection gene for 
        each edge - or connection - in the graph. It contains information about which two nodes are 
        connected, which weight the connection has its innovation number and wether or not the gene is 
        "enabled". The meaning of the last two attributes will be explained in a second.
        """

        conn_gene_title = CText("Connection Genes").scale(0.25)
        grid["conn_gene_title"] = conn_gene_title

        conn_gene_title.move_to(starting_position + DOWN*(conn_gene_title.height/2 + grid_padding) + RIGHT*(conn_gene_title.width/2))
        self.play(Write(conn_gene_title), run_time=1)
        starting_position = starting_position + DOWN*(conn_gene_title.height + grid_padding*2)

        max_cols = 5
        for i, conn_gene in enumerate(conn_genes_vis):
            right_shift = (i % max_cols) * conn_gene.width + (i % max_cols) * grid_padding
            down_shift = (i // max_cols) * conn_gene.height + (i // max_cols) * grid_padding
            conn_gene.move_to(
                starting_position + 
                DOWN*(conn_gene.height/2) + DOWN*down_shift + 
                RIGHT*(conn_gene.width/2) + RIGHT*right_shift
                )
            self.play(Create(conn_gene), run_time=0.1)

        self.wait(2)

        """
        But first: Some terminology. The graph representation as well as the 
        corresponding behaviour of the network are described as its phenotype, whereas its 
        node and connection genes form the genotype.
        """

        phenotype_title = CText("Phenotype", weight=BOLD).scale(0.4)
        phenotype_title.move_to(UP*(self.camera.frame_height/2 - 0.5 - phenotype_title.height/2) + LEFT*(self.camera.frame_width/2 - h_margin - phenotype_title.width/2))
        genotype_title = CText("Genotype", weight=BOLD).scale(0.4)
        genotype_title.move_to(UP*(self.camera.frame_height/2 - 0.5 - genotype_title.height/2) + LEFT*(self.camera.frame_width/2 - cell.width - h_margin*3 - genotype_title.width/2))
        
        self.play(Write(phenotype_title), Write(genotype_title), run_time=1)

        grid.update({
            "phenotype_title": phenotype_title,
            "genotype_title": genotype_title
        })
        
        """
        =========== MUTATION ===========
        Now we will have a look at the process mutation and how it affects the genotype and phenotype of the network.
        The first and most likely form of mutation is a change in the weight of a connection. 
        """

        self.play(
            cell.membrane.animate.set_stroke(color=color_mutation),
            run_time=2
        )

        # Setup
        weight_increase = 3
        mutation_duration = 60*2  # 2 secs á 60 fps
        weight_increment = weight_increase / mutation_duration
        weight = conn_genes_vis[1][0][2]
        def weight_updater(mobject, dt):
            new_val = mobject.get_value() + weight_increment
            mobject.set_value(new_val)
            cell.nucleus.connection_genes.iloc[1, 3] = new_val
            cell.nucleus.adjust_edge_opacity()

        # Highlight edge + text and set membrane to signiture mutation color
        self.play(
            cell.nucleus.edges[(0, 10)].animate.set_color(color_highlight),
            weight.animate.set_color(color_highlight),
            run_time=1
            )

        # Animate weight change
        weight[1].add_updater(weight_updater)
        self.wait(2) # anim35
        weight[1].remove_updater(weight_updater)

        # Remove highlighting
        self.play( # anim36
            cell.nucleus.edges[(0, 10)].animate.set_color(WHITE),
            weight.animate.set_color(WHITE),
            run_time=1
            )        

        self.wait(2)

        """
        The second possible mutation is the addition of a new connection. In our case there is now a new edge connecting node 6 and 10.
        """

        cell.nucleus.add_edge(6, 10, 2.9, False)
        cell.nucleus.adjust_edge_opacity()

        new_conn_gene = cell.nucleus.get_visual_genotype(font_scaling=grid_font_scaling)["connection_genes"][-1]
        grid["conn_genes"].add(new_conn_gene)
        
        i += 1
        right_shift = (i % max_cols) * conn_gene.width + (i % max_cols) * grid_padding
        down_shift = (i // max_cols) * conn_gene.height + (i // max_cols) * grid_padding
        new_conn_gene.move_to(
            starting_position + 
            DOWN*(conn_gene.height/2) + DOWN*down_shift + 
            RIGHT*(conn_gene.width/2) + RIGHT*right_shift
            )

        cell.nucleus.edges[(6, 10)].set_color(color_highlight),
        new_conn_gene[-1].set_stroke(color=color_highlight),   

        self.play(
            GrowFromCenter(cell.nucleus.edges[(6, 10)]),
            GrowFromCenter(new_conn_gene),
            run_time=2
            )

        self.play(
            cell.nucleus.edges[(6, 10)].animate.set_color(WHITE),
            new_conn_gene[-1].animate.set_stroke(color=WHITE),
            run_time=2
        )

        self.wait(2)

        """
        The last possibility is of course the addition of a new node. This one is by far the least likely to occur, because
        the NEAT algorithm tries to find the least complex solution. Therefore weight changes occur much more often than node mutations.
        """

        # Create a new nucleus containing the new node for phenotype
        node_genes, connection_genes = cell.nucleus.node_genes, cell.nucleus.connection_genes
        node_genes.iloc[9, 1] = 2  # change node_level of output nodes to 2
        node_genes.iloc[10, 1] = 2
        node_genes = node_genes.append({
            "innovation_number": 11,
            "node_level": 1
        }, ignore_index=True)
        new_nucleus = Nucleus(cell=cell, genes=(node_genes, connection_genes))
        new_nucleus.move_to(cell.nucleus.get_center())
        

        # Create the node gene vis box of the new node for genotype
        new_node_genes = new_nucleus.get_visual_genotype(font_scaling=grid_font_scaling)["node_genes"] 
        new_node_gene = [node_gene for node_gene in new_node_genes if node_gene[0][1].text[:11] == "HiddenLayer"][0]
        grid["hidden_layer"] = VGroup(new_node_gene)
        new_node_gene.move_to(grid["input_layer"][0][1].get_corner(DOWN + LEFT) + RIGHT*(new_node_gene.width/2) + DOWN*(grid_padding + new_node_gene.height/2))


        # Create two new connections for the connection that gets split. Delete the old one.
        new_nucleus.add_edge(6, 11, 1.0, False)
        new_nucleus.add_edge(11, 10, 1.0, False)
        del new_nucleus.edges[(6, 10)]
        cell.nucleus.adjust_edge_opacity()

        new_nucleus.edges[(6, 11)].set_color(color_highlight),
        new_nucleus.edges[(11, 10)].set_color(color_highlight),


        # Create two connection gene vis boxes for the new connections      
        new_conn_genes = new_nucleus.get_visual_genotype(font_scaling=grid_font_scaling)["connection_genes"][-2:]
        grid["conn_genes"].add(*new_conn_genes)
        
        for new_conn_gene in new_conn_genes:
            i += 1
            right_shift = (i % max_cols) * new_conn_gene.width + (i % max_cols) * grid_padding
            down_shift = (i // max_cols) * new_conn_gene.height + (i // max_cols) * grid_padding
            new_conn_gene.move_to(
                starting_position + 
                DOWN*(new_conn_gene.height/2 + down_shift) +
                RIGHT*(new_conn_gene.width/2 + right_shift)
            )
            new_conn_gene[-1].set_stroke(color=color_highlight),
   

        # Create text "disabled" for the old conn gene vis box
        disabled_text = CText("Disabled", color=RED).scale(grid_font_scaling)
        disabled_text.move_to(
            grid["conn_genes"][2][0][3].get_corner(UP + LEFT) + 
            RIGHT*disabled_text.width/2 + 
            DOWN*(disabled_text.height/2 + output_layer[0].height + grid_padding)
            )


        self.play(
            # Make space in grid
            VGroup(
                grid["output_layer"], 
                grid["conn_gene_title"], 
                grid["conn_genes"],
                ).animate.shift(DOWN*(output_layer[0].height + grid_padding)),
            
            run_time=2,
        )

        self.play(
            # Add node to phenotype
            ReplacementTransform(
                mobject=VGroup(*cell.nucleus.vertices.values())[:9],
                target_mobject=VGroup(*new_nucleus.vertices.values())[:9],
            ),
            ReplacementTransform(
                mobject=VGroup(*cell.nucleus.vertices.values())[9:11],
                target_mobject=VGroup(*new_nucleus.vertices.values())[9:11],
            ),
            ReplacementTransform(
                mobject=VGroup(*cell.nucleus.edges.values())[:2],
                target_mobject=VGroup(*new_nucleus.edges.values())[:2],
            ),
            Create(new_nucleus.vertices[11]),
            FadeIn(new_nucleus.edges[(6, 11)]),
            FadeIn(new_nucleus.edges[(11, 10)]),
            FadeOut(cell.nucleus.edges[(6, 10)]),
            
            # Insert new node gene in grid
            FadeIn(new_node_gene),

            # Insert new conn genes in grid
            FadeIn(new_conn_genes[0]),
            FadeIn(new_conn_genes[1]),

            run_time=3
        )

        # replace old nucleus with new one
        cell.nucleus = new_nucleus   
        grid["conn_genes"][2][0][3] = disabled_text

        # set color back to white
        self.play(
            new_nucleus.edges[(6, 11)].animate.set_color(WHITE),
            new_nucleus.edges[(11, 10)].animate.set_color(WHITE),
            new_conn_genes[0][-1].animate.set_stroke(color=WHITE),
            new_conn_genes[1][-1].animate.set_stroke(color=WHITE),
            cell.membrane.animate.set_stroke(color=WHITE),
            run_time=3
        )

        self.wait(2)

   
        """
        ========== Speciation ============
        Now that we understand how new genes are introduced into the genotype, lets zoom out
        and look at a small population of size 4, where each network was affected by different mutations.        
        """

        species_list = read_file(generation=0, run=run, file="species")
        basis_node_genes, basis_conn_genes = decypher_genotype(species_list[0].genotypes[0])

        cell = VGroup(cell.membrane, cell.nucleus)  

        # Create genotype for cell 2
        cell2_node_genes = pd.concat([basis_node_genes, pd.DataFrame({
            "innovation_number": [11, 12],
            "node_level": [1, 1]
        })])        
        cell2_node_genes.iloc[9, 1] = 2
        cell2_node_genes.iloc[10, 1] = 2

        cell2_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [6, 8, 9, 13, 15, 16],
            "in_node": [3, 3, 12, 4, 4, 12],
            "out_node": [10, 12, 10, 9, 12, 9],
            "weight": [0.4, 0.6, 1.5, 0.2, 2.4, 1.9],
            "is_disabled": [True, False, False, True, False, False],
        })])

        # Create genotype for cell 3
        cell3_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [3, 10, 12, 11],
            "in_node": [6, 8, 1, 4],
            "out_node": [10, 10, 9, 10],
            "weight": [1.3, 0.3, 1.2, 2.4],
            "is_disabled": [False, False, False, False],
        })])

        # Create genotype for cell 4
        cell4_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [7, 11, 3, 14],
            "in_node": [2, 4, 6, 5],
            "out_node": [10, 10, 10, 9],
            "weight": [0.5, 1.4, 2.0, 0.9],
            "is_disabled": [False, False, False, False],
        })])

        cell2 = Cell(width=cell_height/2, height=cell_height, genes=(cell2_node_genes, cell2_conn_genes), num_waves=6, wobble_frequency=3)
        cell2.move_to((10, 6, 0)).rotate_all(PI/0.4)
        cell3 = Cell(width=3, height=3, genes=(basis_node_genes, cell3_conn_genes), num_waves=8, wobble_frequency=3)
        cell3.move_to((8, -5, 0))
        cell4 = Cell(width=3, height=3, genes=(basis_node_genes, cell4_conn_genes), num_waves=8, wobble_frequency=3)
        cell4.move_to((-9, -3.5, 0))
        self.add(cell2, cell3, cell4)

        # von 16 auf 7 war der
        # 7:12.4
        # 16:28.4
        self.play(
            FadeOut(VGroup(*grid.values())),
            )
        
        self.play(
            self.camera.frame.animate.scale_to_fit_height(16),
            cell.animate.shift(UP),
            run_time=4
        )





class SceneFullSimulation(Scene00Setup):
    def construct(self):
        super().construct()

        margin = 1

        # Create Map
        map = Rectangle(height=4, width=4*(16/9), color=WHITE)
        map.move_to(DOWN*(self.camera.frame_height/2 - map.height/2 - margin) + LEFT*(self.camera.frame_width/2 - map.width/2 - margin))
        map_specs = [map.get_start(), map.get_end(), map.get_top(), map.get_bottom()]

        # Game Snipped
        game = Rectangle(height=4, width=4*(16/9), color=WHITE)
        game.move_to(DOWN*(self.camera.frame_height/2 - game.height/2 - margin) + RIGHT*(self.camera.frame_width/2 - game.width/2 - margin))

        # Create Gen Counter
        gen_counter = CText("Generation n").scale(0.5).move(UP*(self.camera.frame_height/2 - gen_counter.height/2 - margin) + LEFT*(self.camera.frame_width/2 - gen_counter.width/2 - margin))

        # Create Phase Ticker
        phase_ticker = CText("Setup").scale(0.5).move(UP*(self.camera.frame_height/2 - phase_ticker.height/2 - margin) + RIGHT*(self.camera.frame_width/2 - phase_ticker.width/2 - margin))

        self.add(game, map, gen_counter, phase_ticker)
        

        last_gen = 130
        for gen in range(last_gen + 1):
            # Update texts
            gen_counter.text = f"Generation {gen}"
            gen_counter.move_to(UP*(self.camera.frame_height/2 - gen_counter.height/2 - margin) + LEFT*(self.camera.frame_width/2 - gen_counter.width/2 - margin))
            phase_ticker.text = "Initialisation"
            phase_ticker.move_to(UP*(self.camera.frame_height/2 - phase_ticker.height/2 - margin) + RIGHT*(self.camera.frame_width/2 - phase_ticker.width/2 - margin))

            species_all = read_file(generation=gen, run=run, file="species")
            num_species = len(species_all)

            # Get 4 Genotype samples from different species if possible
            genotype_samples = []
            sample_distribution = {
                1: [4],
                2: [2, 2],
                3: [1, 1, 2],
                4: [1, 1, 1, 1]
            }

            if num_species > 4:
                species_samples = random.sample(range(num_species), 4)
                for i in species_samples:
                    genotype_samples.extend(random.sample(species_all[i].genotypes, 1))
            else:
                distribution = sample_distribution.get(num_species, [1] * num_species)
                for i, num_samples in enumerate(distribution):
                    genotype_samples.extend(random.sample(species_all[i].genotypes, num_samples))

            # Get the remanining Genotypes
            remaining_genotypes = [genotype for genotype in species_all for species in species_all if genotype not in genotype_samples]

            # Create Cell Samples
            for i, genotype in enumerate(genotype_samples):
                genes = decypher_genotype(genotype)
                cell_height = 4
                cell = Cell(width=cell_height/2, height=cell_height, genes=genes, color=WHITE, num_waves=6, wobble_frequency=3, opacity=0)

            # zellen positionieren. Einfach manuell
                
            # alle zellen gleichzeitig spawnen. Ne farbe wählen fürs aufpoppen neuer zellen wählen.
            self.play(
                cell.membrane.animate.set_opacity(1),
                cell.nucleus.animate.set_opacity(0.2),
                run_time=1
                )

            # Erstmal auf random movement der zellen verzichten. Kann man später auch noch hinzufügen fall zeit passen sollte


            if gen == 5:
                # TODO: Speciation erklären. Hier zum ersten mal 2 Spezien
                pass

            if gen == "x": 
                # TODO: Auf neuen run von amos warten und schauen wo interessantes behaviour ist
                pass

            if gen == "y": 
                # TODO: Animation einmal ohne tweaks laufen lassen und schauen wo man mutation, crossover etc. erklären kann
                pass


class SceneForExperiments(Scene00Setup):
    def construct(self):
        ld = LabeledDot("01")

        # self.add(ld)

        # self.wait(1)
        # ld.set_style(
        #     fill_color=WHITE,

        #     # stroke_color=RED,
        #     # stroke_width=2,
        #     # stroke_opacity=1,

        #     background_stroke_color=RED,
        #     background_stroke_opacity=1,
        #     background_stroke_width=3,

        #     # fill_opacity=0.8,
        #     # sheen_factor=1,
        # )
        # # print(ld.get_style())
        # # # ld.set_stroke(color=RED)
        # # ld.set_fill(color=YELLOW_E)
        # # # ld.set_background_stroke(color=PINK)
        # # ld.set_opacity(0.8)
        

        # self.wait(1)

        
        species_list = read_file(generation=0, run=run, file="species")
        genes = decypher_genotype(species_list[0].genotypes[0])
        cell_height = 4
        cell = Cell(width=cell_height/2, height=cell_height, genes=genes, num_waves=6, wobble_frequency=3)
        cell = VGroup(cell.membrane, cell.nucleus)

        angle = PI*1-4
        direction = np.array([-np.cos(angle), np.sin(angle), 0])

        self.play(
            Rotate(cell, angle),
        )

        self.play(
            cell.animate.move_to(direction*2),
        )



