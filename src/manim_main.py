import os
from manim import *
from manim_utils import read_file, decypher_genotype
from manim_utils import CText
from manim_mobjects import Cell
import random

# Debugging
render_animations = True
random.seed = 42
short_run_time = 0.2
long_run_time = 1

# Neat
run = '2023_12_07_19_59_20_340927'

class Scene00Setup(MovingCameraScene):
    def play(self, *args, **kwargs):
        if render_animations:
            super().play(*args, **kwargs)

    def construct(self):
        self.camera.background_color = "#222222"
        self.initial_height = self.camera.frame_height
        self.initial_width = self.camera.frame_width

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
        cell = Cell(width=cell_height/2, height=cell_height, genes=genes, color=WHITE, num_waves=6, wobble_frequency=3, opacity=0)
        self.play(
            cell.membrane.animate.set_opacity(1),
            cell.nucleus.animate.set_opacity(0.2),
            run_time=short_run_time
            )

        # Zoom in
        v_margin, h_margin = 1.5, 0.5
        self.play(self.camera.frame.animate.scale_to_fit_height(cell_height + v_margin*2), run_time=short_run_time)

        # self.play(self.camera.resize_frame_shape(height=cell.height + 1), run_time=short_run_time)
        # self.play(self.camera.auto_zoom([cell], margin=0.5), run_time=short_run_time)

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
                self.play(cell.nucleus[vertice].animate.set_color(YELLOW), run_time=short_run_time) # run_time=0.15
                cell.nucleus[vertice].set_color(WHITE)
        
        self.wait(short_run_time)

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
                self.play(cell.nucleus[vertice].animate.set_color(YELLOW), run_time=short_run_time)
                cell.nucleus[vertice].set_color(WHITE)
        
        self.wait(short_run_time)

        """
        This most basic neural network there are only 2 connections, both connecting one of the outputs 
        with the bias node.  meaning that the inputs aren't influencing the actions of the lunar lander at all. 
        This is obviously a pretty poor architecture, since the gamestate is in no way contributing to the 
        actions that the lunar lander takes, meaning that information like the landers coordinates and velocity 
        are entirely disregarded.
        """

        # Highlight connections
        for i, edge in enumerate(cell.nucleus.edges):
            self.play(cell.nucleus.edges[edge].animate.set_color(YELLOW), run_time=short_run_time)
            cell.nucleus.edges[edge].set_color(WHITE)
        
        self.wait(short_run_time)

        """
        This issue will soon be adressed by the process of mutation. But before we look into that, 
        we must understand how the current architecture of the network is expressed in the context 
        of the NEAT algorithm. 
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
            cell.nucleus.animate.set_opacity(1),
            run_time = short_run_time
            )

        self.wait(short_run_time)

        """
        First we look at its node gene. There is a node gene for each node of 
        the network expressing which layer it belongs to. 
        """
        genes_vis_scaling = 0.15
        node_genes_vis, conn_genes_vis = cell.nucleus.get_visual_genotype(scale_factor=genes_vis_scaling).values()

        starting_position = UP*(self.camera.frame_height/2 - v_margin) + LEFT*(self.camera.frame_width/2 - cell.width - h_margin*3)
        grid_padding = 0.2

        node_gene_title = CText("Node Genes").scale(0.25)
        node_gene_title.move_to(starting_position + DOWN*(node_gene_title.height/2) + RIGHT*(node_gene_title.width/2))
        self.play(Write(node_gene_title), run_time=short_run_time)
        starting_position = starting_position + DOWN*(node_gene_title.height + grid_padding)
        
        input_layer = [node_gene for node_gene in node_genes_vis if node_gene[0][1].text == "InputLayer"]
        hidden_layer = [node_gene for node_gene in node_genes_vis if node_gene[0][1].text[:11] == "HiddenLayer"]
        output_layer = [node_gene for node_gene in node_genes_vis if node_gene[0][1].text == "OutputLayer"]

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

        conn_gene_title.move_to(starting_position + DOWN*(conn_gene_title.height/2 + grid_padding) + RIGHT*(conn_gene_title.width/2))
        self.play(Write(conn_gene_title), run_time=short_run_time)
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
        # starting_position = starting_position + DOWN*((conn_gene.__len__()//6 + 1) * conn_gene[0].height + (conn_gene.__len__()//6 + 1) * grid_padding)

        """
        But first: Some terminology. The graph representation as well as the 
        corresponding behaviour of the network are described as its phenotype, whereas its 
        node and connection genes form the genotype.
        """

        grid = VGroup(node_gene_title, conn_gene_title, *node_genes_vis, *conn_genes_vis)

        phenotype_title = CText("Phenotype", weight=BOLD).scale(0.4)
        phenotype_title.move_to(UP*(self.camera.frame_height/2 - 0.5 - phenotype_title.height/2) + LEFT*(self.camera.frame_width/2 - h_margin - phenotype_title.width/2))
        genotype_title = CText("Genotype", weight=BOLD).scale(0.4)
        genotype_title.move_to(UP*(self.camera.frame_height/2 - 0.5 - genotype_title.height/2) + LEFT*(self.camera.frame_width/2 - cell.width - h_margin*3 - genotype_title.width/2))
        
        self.play(Write(phenotype_title), Write(genotype_title), run_time=short_run_time)
        
        """
        Now we will have a look at the process mutation and how it affects the genotype and phenotype of the network.
        The first and most likely form of mutation is a change in the weight of a connection.
        """
        conn_genes_df = cell.nucleus.connection_genes
        conn_genes_df[conn_genes_df["innovation_number"] == 2]["weight"] = 1.24 # in: 0, out: 10
        conn_genes_vis_new = cell.nucleus.get_visual_genotype(scale_factor=genes_vis_scaling)["connection_genes"]

        edge = (0, 10)
        old_conn_gene = conn_genes_vis[1][0]  # 1: second conn gene vis, 0: Text, not Rectangle
        new_conn_gene = conn_genes_vis_new[1][0]

        self.play(
            old_conn_gene[2].animate.set_color(YELLOW), 
            cell.nucleus.edges[edge].animate.set_color(YELLOW),
            cell.nucleus.edges[edge].animate.set_opacity(1),
            run_time=1
            )
        
        self.play(
            Transform(old_conn_gene, new_conn_gene),
        )


        cell.nucleus.connection_genes[cell.nucleus.connection_genes["innovation_number"] == 2]["weight"] = 1.24
        new_connection_genes = cell.nucleus.get_visual_genotype(scale_factor=genes_vis_scaling)["connection_genes"]
        
        
        conn_genes_vis[-1][0][2] = new_connection_genes[-1][0][2]
        

        
        init_num = float(weight.text[-4:])


        new_weight_txt = Text(f"Weight {1.24}").scale(0.25).move_to(weight_txt.get_center())
        self.play(Transform(weight_txt, new_weight_txt))
        conn_genes_vis[-1][0][2] = new_weight_txt

        self.wait(0.5)

        weight.set_color(WHITE)
        cell.nucleus.edges[(0, 10)].set_color(WHITE)
        self.wait(1)


        """
        The second possible mutation is the addition of a new connection.
        """

        cell.nucleus.add_edge(6, 10, 0.5, False)
        cell.nucleus.adjust_edge_opacity()

        node_genes_vis, conn_genes_vis = cell.nucleus.get_visual_genotype(scale_factor=0.15).values()
        new_conn_gene = conn_genes_vis[-1]
        
        i += 1
        right_shift = (i % max_cols) * conn_gene.width + (i % max_cols) * grid_padding
        down_shift = (i // max_cols) * conn_gene.height + (i // max_cols) * grid_padding
        new_conn_gene.move_to(
            starting_position + 
            DOWN*(conn_gene.height/2) + DOWN*down_shift + 
            RIGHT*(conn_gene.width/2) + RIGHT*right_shift
            )
        
        self.add(cell.nucleus.edges[(6, 10)], new_conn_gene)
        self.play(
            cell.nucleus.edges[(6, 10)].animate.set_color(YELLOW),
            new_conn_gene[-1].animate.set_stroke(YELLOW),
            run_time=long_run_time
            )
        cell.nucleus.edges[(6, 10)].set_color(WHITE)
        new_conn_gene[-1].set_stroke(WHITE)



        self.wait(2)


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
                run_time=short_run_time
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
        rect = Rectangle(color=RED).shift(RIGHT)
        circle = Circle(color=BLUE).shift(LEFT)
        num = float(0.55)
        txt = Text(f"{num}").shift(DOWN)
        txt2 = Text("aah").shift(UP)

        self.add(rect, circle, txt, txt2)
        self.wait(1)
        rect.set_color(GREEN)  # Use set_color instead of direct assignment
        circle.set_color(YELLOW)
        txt2.text = "bbbb"

        def txt_update(txt, dt):
            new_txt = Text(f"{float(num + 1 * dt):.2f}")
            self.play(Transform(txt, new_txt))
            txt = new_txt

        txt.add_updater(txt_update)
        self.wait(3)
        txt.remove_updater(txt_update)

        self.wait(1)