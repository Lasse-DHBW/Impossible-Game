from classes.manim_mobjects import Cell, Nucleus, ColorChangeAnimation, Particle, Closeup
from classes.manim_utils import read_file, decypher_genotype, calc_lag_ratio, CText
from manim import *
import pandas as pd
import random


# Debugging
random.seed(42)

# Color schema
colors = {
    "highlight": "#EBA607",  # Yellow
    "mutation": "#305FB0", # Blue
    "elimination": "#CF2D24", # Red
    "crossover": "#20944F", # Green
    "speciation": "#892EA4", # Purple
    "white": GRAY_A,
}

# Neat
run = "20240104_225458"


class Scene00Setup(MovingCameraScene):
    # Scene for global settings that can be inherited by all other scenes
    def construct(self):
        self.camera.background_color = "#222222"
        self.camera.frame.scale_to_fit_height(16)
        config["frame_rate"] = 50


class Scene03Individual(Scene00Setup):
    # First half of the video, where the theoretical foundations are layed out
    def construct(self):
        super().construct()

        """ ========== Cold Opener ============
        1 | -n 0,9 | voice: 20 sec, anim: 21 sec
        Imagine the dawn of life on our planet, when simple single-celled organisms first started to emerge - 
        each one a bundle of potential, mutating and propagating its genetic code. Over time this process of 
        natural selection yielded great genetic diversity and ultimately - intelligent life. 
        """

        frame_x_radius = self.camera.frame.get_width() / 2
        frame_y_radius = self.camera.frame.get_height() / 2

        # Spawn background particles (Class defined in manim_mobjects)
        particles = VGroup()
        for _ in range(50):
            scale_factor = random.uniform(0.5, 3)
            particle = Particle(scale_factor, width=0.02*scale_factor, height=0.03*scale_factor, color=WHITE, stroke_opacity=0.3*scale_factor, stroke_width=1)
            particles.add(particle)

        # Setup updater for particle movement
        def update_particles(particles, dt):
            for particle in particles:
                particle.shift(particle.velocity * dt)  # dt is used here
                # Bounce off walls logic
                if not (-frame_x_radius < particle.get_x() < frame_x_radius):
                    particle.velocity[0] *= -1
                if not (-frame_y_radius < particle.get_y() < frame_y_radius):
                    particle.velocity[1] *= -1

        # Randomly place particles and add the updater to each of them
        [particle.move_to(random.uniform(-frame_x_radius, frame_x_radius)*RIGHT + random.uniform(-frame_y_radius, frame_y_radius)*UP).rotate(random.uniform(0,PI*2)) for particle in particles]
        [particle.add_updater(update_particles) for particle in particles]

        self.add(particles)

        # Create some cells to populate foreground.
        genes = []
        for gen in [11, 52, 191, 222, 348]:
            # To avoid manual configuration their genes are taken from individuals in the simulation, which will be shown later
            species_list = read_file(generation=gen, run=run, file="species")
            genes.append(decypher_genotype(species_list[0].genotypes[0]))

        c1 = Cell(width=2, height=4, genes=genes[0], fitness=-500, num_waves=4, wobble_frequency=3).move_to((-10, -2, 0)).rotate(PI*0.12)
        c2 = Cell(width=1.5, height=3, genes=genes[1], fitness=-500, num_waves=4, wobble_frequency=1).move_to((-7, 6, 0)).rotate(PI*0.75)
        c3 = Cell(width=3, height=6, genes=genes[2], fitness=-500, num_waves=7, wobble_frequency=6).move_to((2, 5, 0)).rotate(PI*0.41)
        c4 = Cell(width=2.5, height=5, genes=genes[3], fitness=-500, num_waves=6, wobble_frequency=2).move_to((-5, -3, 0)).rotate(PI*0.2) #d
        c5 = Cell(width=4, height=8, genes=genes[4], fitness=-500, num_waves=8, wobble_frequency=4).move_to((8, -3, 0)).rotate(PI*1.7)
        c1.membrane.set_stroke(opacity=0.5)
        c2.membrane.set_stroke(opacity=0.3)
        c3.membrane.set_stroke(opacity=0.8)
        c4.membrane.set_stroke(opacity=0.6)
        c5.membrane.set_stroke(opacity=1)
        [c.nucleus.set_style(fill_opacity=0, stroke_opacity=0) for c in [c1, c2, c3, c4, c5]]

        # Fetch protagonist genes
        species_list = read_file(generation=0, run=run, file="species")
        genes = decypher_genotype(species_list[0].genotypes[0])
        
        # Create protagonist cell
        cell_height = 4
        cell = Cell(width=cell_height/2, height=cell_height, genes=genes, fitness=-500, num_waves=5, wobble_frequency=4)
        cell.nucleus.set_style(fill_opacity=0, stroke_opacity=0)

        # Spawn cells
        self.play(LaggedStart(
            *[FadeIn(cell) for cell in [c1, c4, c2, c3, c5]],
            lag_ratio=calc_lag_ratio(5, 3, 5),
            run_time=5,
        ))
        
        # Add some 
        self.wait(1)
        self.play(c2.membrane.animate.set_stroke(color=colors["mutation"]), run_time=2)
        self.wait(1)
        self.play(
            c5.membrane.animate.set_stroke(color=colors["mutation"]),
            c2.membrane.animate.set_stroke(color=WHITE),
            run_time=2,
            )

        self.wait(1)
        self.play(
            c5.membrane.animate.set_stroke(color=colors["white"]),
            c1.membrane.animate.set_stroke(color=colors["crossover"]),
            c4.membrane.animate.set_stroke(color=colors["crossover"]),
            run_time=3
        )
        self.play( 
            FadeIn(cell),
            run_time=3
        )
        self.play(
            c1.membrane.animate.set_stroke(color=colors["white"]),
            c4.membrane.animate.set_stroke(color=colors["white"]),
            run_time=2
        )
        self.wait(1)

        """
        2 | -n 10,12 | voice: 7 sec, anim: 8 sec
        The NEAT algorithm aims to mimic this process in order to train artificial neural networks to solve specific problems. 
        """

        # Fade in the neural networks (nucleus) of the cells
        self.wait(1)
        self.play(
            # Use fill_opacity and stroke_opacity! Cant use nucleus.set_opacity because of LabeledDot.background_stroke_opacity
            *[c.nucleus.animate.set_style(fill_opacity=0.75, stroke_opacity=0.75) for c in [cell, c1, c2, c3, c4, c5]],
            run_time=3
        )
        self.wait(5)

        """
        3 | -n 13,16 | voice: 9 sec, anim: 10 sec
        In our case, the networks will learn to play 'lunar lander', an old arcade game where the objective is to land a rocket on the surface of the moon.
        """

        # Display Box on the top right corner of the screen. The lunar lander clip will be inserted in post production 
        game_snipped = Rectangle(height=4.21875, width=6.875, color=colors['highlight']).move_to((14.2-1-5.5/2, 8-1-3.375/2, 0)).set_stroke(width=8)
        self.play(FadeIn(game_snipped), run_time=2)
        self.wait(3) # Clip lasts 5 secs - subtract 2 to account for the fade in and out
        self.play(FadeOut(game_snipped), run_time=2)
        self.wait(3)

        """
        4 | -n 17,18  | voice 15 sec, anim: 15
        But how exactly does it work? Let's dive in and find out. 
        We'll start with the simplest possible architecture for the task at hand - A neural net consisting of 9 input and 2 output nodes.
        """

        # Zoom in to protagonist cell
        v_margin, h_margin = 1.5, 0.5
        self.play( 
            self.camera.frame.animate.scale_to_fit_height(cell_height + v_margin*2), 
            *[FadeOut(cell) for cell in [c1, c2, c3, c4, c5]],
            FadeOut(particles),
            run_time=3
            )
        self.wait(12)

        """
        5 | -n 19,28 | voice: 10 sec, anim: 10 sec
        The inputs will be the aircrafts position, velocity, angle, and contact sensors, plus an additional bias node, always set to one.
        """

        # Highlight input nodes
        for vertice in cell.nucleus.vertices:
            if vertice in cell.nucleus.partitions[0]:
                cell.nucleus[vertice].set_background_stroke(opacity=1)
                cell.nucleus[vertice].set_fill(opacity=1)
                self.wait(0.5) # 4.5 secs
                cell.nucleus[vertice].set_background_stroke(opacity=0)

        self.wait(5.5)

        """
        6 | -n 29,32 | voice: 7 sec, anim: 8 sec
        The networks output nodes are few but crucial, representing the aircrafts vertical and horizontal thrust. 
        """

        self.wait(5.5)
        # Highlight output nodes
        for vertice in cell.nucleus.vertices:
            if vertice in cell.nucleus.partitions[1]:
                cell.nucleus[vertice].set_background_stroke(opacity=1)
                cell.nucleus[vertice].set_fill(opacity=1)
                self.wait(0.5) # 1 sec
                cell.nucleus[vertice].set_background_stroke(opacity=0)

        self.wait(1.5)

        """
        7 | -n 33,36 | voice: 22 sec, anim: 22 sec
        In this most basic neural network the only two edges connect the outputs with the bias node.
        Since the bias node is just a constant value of 1, the aircraft actions are entirely independent of its environment.
        This is obviously a pretty poor choice of architecture. An issue that will be adressed once the cells start evolving.
        """

        self.wait(2)
        # Highlight connections
        for edge in cell.nucleus.edges.values():
            edge.set_color(colors["highlight"])
            edge.set_stroke(opacity=1)
            self.wait(3)
            edge.set_color(WHITE)
        
        self.wait(14)

        """ ========== Genotype Terminology ============
        8 | -n 37,39 | voice: 8 sec, anim: 8 sec
        But before we look into that, we must understand how the current architecture of the network is expressed within the NEAT framework. 
        """

        # Move to side
        new_cell_pos = LEFT*(self.camera.frame_width / 2 - cell.width / 2 - h_margin) 
        self.play( 
            cell.animate.move_to(new_cell_pos),
            run_time=3
            )       

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
            run_time = 4
            )

        self.wait(1) 

        """
        9 | -n 40,52 | voice: 7 sec, anim: 7.5
        The vertices of the network are represented by so-called 'node genes'. They indicate which layer a given node belongs to.
        """

        grid = {"divider": divider}
        grid_font_scaling = 0.15
        grid_padding = 0.2
        
        # Create a VGroup for each node gene via .get_visual_genotype() to visualize the underlying dataframe
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
                self.play(
                    FadeIn(node_gene[0]), 
                    FadeIn(node_gene[1]),
                    run_time=0.5) # 11 iterations

            starting_position = starting_position + DOWN*(layer[0].height + grid_padding)

        self.wait(1)

        """
        10 | -n 53,56 | voice: 31 sec, anim: 
        Additionally there is one connection gene for each edge in the network. 
        Each of these genes tells us which nodes are linked, the strength of their connection, and whether this connection is currently active. 
        Each connection also has a unique 'innovation number'. Think of it like a global patent number which is granted once a new connection is discovered.
        If another network later evolves the same edge, its corresponding connection gene would bear the same innovation number.
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
            self.play(
                FadeIn(conn_gene[0]), 
                FadeIn(conn_gene[1]),
                run_time=1)

        self.wait(29)

        """
        11 | -n 57,61 | voice: 20 sec
        In the context of NEAT, the term 'genotype' refers to the genetic makeup of a neural network - the set of all its node and connection genes.
        The phenotype, on the other hand, is how the network's structure is expressed - the graph representation as well as the corresponding behaviour.
        """

        phenotype_title = CText("Phenotype", weight=BOLD).scale(0.4)
        phenotype_title.move_to(UP*(self.camera.frame_height/2 - 0.5 - phenotype_title.height/2) + LEFT*(self.camera.frame_width/2 - h_margin - phenotype_title.width/2))
        genotype_title = CText("Genotype", weight=BOLD).scale(0.4)
        genotype_title.move_to(UP*(self.camera.frame_height/2 - 0.5 - genotype_title.height/2) + LEFT*(self.camera.frame_width/2 - cell.width - h_margin*3 - genotype_title.width/2))
        self.wait(2)
        self.play(Write(genotype_title), run_time=1)
        self.wait(7.5)
        self.play(Write(phenotype_title), run_time=1)
        self.wait(8.5)

        grid.update({
            "phenotype_title": phenotype_title,
            "genotype_title": genotype_title
        })
        
        """  ========== Mutation ============
        12 | -n 62,66 | voice: 20 sec, anim: 25 sec
        Now that we've got the terminology out of the way, let's see how these networks evolve. 
        Just like in nature, random mutations can result in changes to the genetic code. 
        They can affect the structure of the network in three ways. By far the most common is a change in the weight of a connection.
        """

        self.play(
            cell.membrane.animate.set_stroke(color=colors["mutation"]),
            run_time=3
        )

        # Setup
        weight_increase = 3
        mutation_duration = 50*4 # 2 secs á 50 fps
        weight_increment = weight_increase / mutation_duration
        weight = conn_genes_vis[1][0][2]
        start_opacity = cell.nucleus.edges[(0, 10)].get_style()["stroke_opacity"]
        opacity_increment = (1 - start_opacity) / mutation_duration
        def weight_updater(mobject, dt):
            new_val = mobject.get_value() + weight_increment
            mobject.set_value(new_val)
            cell.nucleus.connection_genes.iloc[1, 3] = new_val
            current_opacity = cell.nucleus.edges[(0, 10)].get_style()["stroke_opacity"]
            cell.nucleus.edges[(0, 10)].set_style(stroke_opacity = current_opacity + opacity_increment)
            # cell.nucleus.adjust_edge_opacity()

        self.wait(13)
        # Highlight edge + text and set membrane to signiture mutation color
        self.play(
            cell.nucleus.edges[(0, 10)].animate.set_color(colors["highlight"]),
            weight.animate.set_color(colors["highlight"]),
            run_time=3
            )

        # Animate weight change
        weight[1].add_updater(weight_updater)
        self.wait(4) 
        weight[1].remove_updater(weight_updater)

        # Remove highlighting
        self.play( 
            cell.nucleus.edges[(0, 10)].animate.set_color(colors["white"]),
            weight.animate.set_color(colors["white"]),
            run_time=2
            )        

        # Manually set the end result of 3.11 again, cause it gets reset for some unknown reason
        weight = conn_genes_vis[1][0][2][1].set_value(3.11)
        cell.nucleus.connection_genes.iloc[1, 3] = 3.11

        """
        13 | -n 67,69 | voice: 8 sec, anim: 10 sec
        The second possible mutation is the addition of a new connection like the one forming between nodes 6 and 10.
        """

        cell.nucleus.add_edge(6, 10, 2.9, False)

        new_conn_gene_vis = cell.nucleus.get_visual_genotype(font_scaling=grid_font_scaling)["connection_genes"][-1]
        grid["conn_genes"].add(new_conn_gene_vis)
        
        i += 1
        right_shift = (i % max_cols) * conn_gene.width + (i % max_cols) * grid_padding
        down_shift = (i // max_cols) * conn_gene.height + (i // max_cols) * grid_padding
        new_conn_gene_vis.move_to(
            starting_position + 
            DOWN*(conn_gene.height/2) + DOWN*down_shift + 
            RIGHT*(conn_gene.width/2) + RIGHT*right_shift
            )

        cell.nucleus.edges[(6, 10)].set_opacity(0)
        cell.nucleus.edges[(6, 10)].set_color(colors["highlight"]),
        new_conn_gene_vis[-1].set_stroke(color=colors["highlight"]),   

        self.wait(3)
        self.play(
            cell.nucleus.edges[(6, 10)].animate.set_opacity(1),
            GrowFromCenter(new_conn_gene_vis),
            run_time=5
            )

        self.play(
            cell.nucleus.edges[(6, 10)].animate.set_color(WHITE),
            new_conn_gene_vis[-1].animate.set_stroke(color=WHITE),
            run_time=2
        )


        """
        14 | -n 70,74 | voice: 24 sec, anim: 26 sec
        A less common mutation in the NEAT algorithm is the creation of a new node. 
        During this operation a node is inserted into an existing connection, effectively splitting it into two.
        This mutation happens rarely because the algorithm tries to seek out the most streamlined solution, 
        prioritizing smaller, well-structured networks over bulkier ones.
        """

        # Create a new nucleus containing the new node for phenotype
        node_genes, connection_genes = cell.nucleus.node_genes, cell.nucleus.connection_genes
        node_genes.iloc[9, 1] = 2  # change node_level of output nodes to 2
        node_genes.iloc[10, 1] = 2
        node_genes = pd.concat([node_genes, pd.DataFrame({
            "innovation_number": [11],
            "node_level": [1]
        })], ignore_index=True)
        new_nucleus = Nucleus(2, 4, genes=(node_genes, connection_genes.iloc[:2])) # Dont include 6 - 10 conn which gets split
        new_nucleus.move_to(cell.nucleus.get_center())

        new_nucleus.vertices[11].set_background_stroke(opacity=1)

        # Create the node gene vis box of the new node for genotype
        new_node_genes = new_nucleus.get_visual_genotype(font_scaling=grid_font_scaling)["node_genes"] 
        new_node_gene = [node_gene for node_gene in new_node_genes if node_gene[0][1].text[:11] == "HiddenLayer"][0]
        grid["hidden_layer"] = VGroup(new_node_gene)
        new_node_gene.move_to(grid["input_layer"][0][1].get_corner(DOWN + LEFT) + RIGHT*(new_node_gene.width/2) + DOWN*(grid_padding + new_node_gene.height/2))
        new_node_gene[1].set_stroke(color=colors["highlight"], width=2)

        # Create two new connections for the connection that gets split. Delete the old one.
        new_nucleus.add_edge(6, 11, 1.0, False)
        new_nucleus.add_edge(11, 10, 1.0, False)
        cell.nucleus.adjust_edge_opacity()

        new_nucleus.edges[(6, 11)].set_color(colors["highlight"]),
        new_nucleus.edges[(11, 10)].set_color(colors["highlight"]),

        # Create two connection gene vis boxes for the new connections      
        new_conn_genes = new_nucleus.get_visual_genotype(font_scaling=grid_font_scaling)["connection_genes"][-2:]
        grid["conn_genes"].add(*new_conn_genes)
        
        for new_conn_gene_vis in new_conn_genes:
            i += 1
            right_shift = (i % max_cols) * new_conn_gene_vis.width + (i % max_cols) * grid_padding
            down_shift = (i // max_cols) * new_conn_gene_vis.height + (i // max_cols) * grid_padding
            new_conn_gene_vis.move_to(
                starting_position + 
                DOWN*(new_conn_gene_vis.height/2 + down_shift) +
                RIGHT*(new_conn_gene_vis.width/2 + right_shift)
            )
            new_conn_gene_vis[-1].set_stroke(color=colors["highlight"], width=2)
            new_conn_gene_vis.set_opacity(0)
   
        # Create text "disabled" for the old conn gene vis box
        disabled_text = CText("Disabled", color=RED).scale(grid_font_scaling)
        disabled_text.move_to(
            grid["conn_genes"][2][0][3].get_corner(UP + LEFT) + 
            RIGHT*disabled_text.width/2 + 
            DOWN*(disabled_text.height/2 + output_layer[0].height + grid_padding)
            )

        self.wait(6)
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
            new_conn_genes[0][1].animate.set_stroke(opacity=1),
            new_conn_genes[1][1].animate.set_stroke(opacity=1),
            new_conn_genes[0][0].animate.set_opacity(1),
            new_conn_genes[1][0].animate.set_opacity(1),            

            run_time=6
        )

        # replace old nucleus with new one
        cell -= cell.nucleus
        cell += new_nucleus   
        cell.nucleus = new_nucleus
        grid["conn_genes"][2][0][3] = disabled_text


        # set color back to white
        self.play(
            new_nucleus.edges[(6, 11)].animate.set_color(WHITE),
            new_nucleus.edges[(11, 10)].animate.set_color(WHITE),
            new_conn_genes[0][-1].animate.set_stroke(color=WHITE, width=1),
            new_conn_genes[1][-1].animate.set_stroke(color=WHITE, width=1),
            new_nucleus.vertices[11].animate.set_background_stroke(opacity=0),
            cell.membrane.animate.set_stroke(color=WHITE, opacity=1),
            new_node_gene[1].animate.set_stroke(color=WHITE, width=1),
            run_time=3
        )

        self.wait(9)

   
        """ ========== Speciation ============
        15 | -n 75,77 | voice: 13 sec, anim:  sec
        Now that we understand how new genes are introduced into the gene pool, lets zoom out
        and look at a small population of size 5, where each network was affected by different mutations.        
        """

        species_list = read_file(generation=0, run=run, file="species")
        basis_node_genes, basis_conn_genes = decypher_genotype(species_list[0].genotypes[0])

        cell0 = cell

        # Create genotype for cell 1
        cell1_node_genes = pd.concat([basis_node_genes, pd.DataFrame({
            "innovation_number": [11, 12],
            "node_level": [1, 1]
        })])        
        cell1_node_genes.iloc[9, 1] = 2
        cell1_node_genes.iloc[10, 1] = 2

        cell1_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [6, 8, 9, 13, 15, 16],
            "in_node": [3, 3, 12, 4, 4, 11],
            "out_node": [10, 12, 10, 9, 11, 9],
            "weight": [0.4, 0.6, 1.5, 0.2, 2.4, 1.9],
            "is_disabled": [True, False, False, True, False, False],
        })])

        # Create genotype for cell 2
        cell2_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [3, 10, 12, 11],
            "in_node": [6, 8, 1, 4],
            "out_node": [10, 10, 9, 10],
            "weight": [1.3, 0.3, 1.2, 2.4],
            "is_disabled": [False, False, False, False],
        })])

        # Create genotype for cell 3
        cell3_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [7, 11, 3, 14],
            "in_node": [2, 4, 6, 5],
            "out_node": [10, 10, 10, 9],
            "weight": [0.5, 1.4, 2.0, 0.9],
            "is_disabled": [False, False, False, False],
        })])

        # Create genotype for cell 4
        cell4_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [7, 55, 12],
            "in_node": [2, 1, 1],
            "out_node": [10, 10, 9],
            "weight": [0.5, 1.4, 1.2],
            "is_disabled": [False, False, False],
        })])

        # Zoom out to aspect ratio 13.5 : 24

        cell1 = Cell(width=cell_height/2, height=cell_height, genes=(cell1_node_genes, cell1_conn_genes), fitness=-481, num_waves=8, wobble_frequency=4)
        cell1.move_to((-0.5, 2.7, 0))
        cell2 = Cell(width=3, height=3, genes=(basis_node_genes, cell2_conn_genes), fitness=-283, num_waves=8, wobble_frequency=2)
        cell2.move_to((1.4, -2.5, 0))
        cell3 = Cell(width=3, height=3, genes=(basis_node_genes, cell3_conn_genes), fitness=-330, num_waves=6, wobble_frequency=4)
        cell3.move_to((5.9, -1.2, 0))
        cell4 = Cell(width=2.5, height=2.5, genes=(basis_node_genes, cell4_conn_genes), fitness=-400, num_waves=5, wobble_frequency=3)
        cell4.move_to((-9, 4.9, 0))

        angles = [PI*0.22, PI*0.23, -PI*0.31]
        [cell.rotate(angle) for cell, angle in zip([cell1, cell2, cell3], angles)]

        self.play(FadeOut(VGroup(*grid.values())),
                  run_time=4)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(13.5),
            FadeIn(cell1),
            FadeIn(cell2),
            FadeIn(cell3),
            FadeIn(cell4),
            run_time=4
            )

        self.wait(6)
        
        """
        16 | -n 78,83 | voice: 16 sec, anim: 17 sec
        As can be seen by their phenotypes the population has branched into two separate species. 
        The first has evolved to have a hidden layer, whereas the second species displays a high interconnectivity between the input and output nodes.
        """

        self.wait(6)
        self.play(
            cell0.animate.set_stroke(color=colors["speciation"]),
            cell1.animate.set_stroke(color=colors["speciation"]),     
            run_time=2   
        )
        self.play(
            cell0.animate.set_stroke(color=colors["white"]),
            cell1.animate.set_stroke(color=colors["white"]), 
            run_time=2
        )
        self.play(
            cell2.animate.set_stroke(color=colors["speciation"]),
            cell3.animate.set_stroke(color=colors["speciation"]),   
            cell4.animate.set_stroke(color=colors["speciation"]),  
            run_time=2       
        )
        self.play(
            cell2.animate.set_stroke(color=colors["white"]),
            cell3.animate.set_stroke(color=colors["white"]),     
            cell4.animate.set_stroke(color=colors["white"]),
            run_time=2 
        )
        self.wait(3)


        """ ========== Fitness ============
        17 | -n 84,86 | voice: 19 sec, anim: 20 sec
        Each of these networks has a global fitness value, that expresses their ability to land the aircraft in a controlled and fuel efficient manner.
        The 'adjusted fitness' on the other hand is equal to an individuals global fitness divided by the number of individuals in its species.
        """
        self.wait(1)

        self.play(
            FadeOut(cell0.nucleus),
            cell0.fitness.animate.set_fill(opacity=1),
            
            FadeOut(cell1.nucleus),
            cell1.fitness.animate.set_fill(opacity=1),

            FadeOut(cell2.nucleus),
            cell2.fitness.animate.set_fill(opacity=1),

            FadeOut(cell3.nucleus),
            cell3.fitness.animate.set_fill(opacity=1),

            FadeOut(cell4.nucleus),
            cell4.fitness.animate.set_fill(opacity=1),

            run_time=3
        )

        self.wait(16)

        """ ========== Elimination ============
        18 | -n 87,90 | voice: 14 sec, anim: 14 sec
        Much like in nature, only the fittest individuals can survive and thus succeed in passing on their genes.
        Therefore each generation, every species looses a certain percentage of their least fit individuals.
        """

        self.wait(8)
        self.play(
            cell4.membrane.animate.set_stroke(color=colors["elimination"], width=2),
            cell0.membrane.animate.set_stroke(color=colors["elimination"], width=2),
            run_time=3,
        )

        self.play(
            FadeOut(cell4.membrane),
            FadeOut(cell4.fitness),
            FadeOut(cell0.membrane),
            FadeOut(cell0.fitness),
            run_time=3,
        )
        self.wait(2)


        """ ============= Cross Over ================
        19 | -n 91,95| voice: 15 sec, anim: 16 sec
        In each species the remaining individuals are then paired up and their genes are combined in a process called "cross over".
        How many offsprings a given species can produce is proportional to the sum of its adjusted fitness values.
        """

        print(cell2.get_center(), cell3.get_center())

        self.play(
            FadeOut(cell2.fitness),
            FadeOut(cell3.fitness),
            FadeOut(cell1.fitness),
            FadeIn(cell2.nucleus),
            FadeIn(cell3.nucleus),
            FadeIn(cell1.nucleus),
            run_time=2,
        )
        cell2.fitness.set_fill(opacity=0)
        cell3.fitness.set_fill(opacity=0)
        cell1.fitness.set_fill(opacity=0)

        self.play(
            cell2.membrane.animate.set_stroke(color=colors["crossover"], width=2),
            cell3.membrane.animate.set_stroke(color=colors["crossover"], width=2),
            run_time=2
        )

        # Cross Over Grid: 3x3 Grid,
        # Frame: 13.5 : 24
        # row height = 3 (cell_height) + 2*0.75 (v_margin) = 4.5
        # col width for cols 1 & 3 = 3 (cell_width) + 2*0.75 (h_margin) = 4.5
        # middle col width = 24 - 2*4.5 = 15
        
        v_margin, h_margin = 0.75, 0.75 

        # Create Grid for cross over explanation
        self.play(
            AnimationGroup(
                FadeOut(cell1),
                run_time=2
            ),
            AnimationGroup(
                cell2.animate.move_to((-9.75, 4.5, 0)),
                cell3.animate.move_to((-9.75, 0, 0)),
                run_time=4
            )
        )

        divider_r1 = DashedLine(
            (-12, 2.25, 0),
            (12, 2.25, 0),
        )
    
        divider_r2 = DashedLine(
            (-12, -2.25, 0),
            (12, -2.25, 0),
        )
        
        divider_c1 = DashedLine(
            (-7.5, 6.75, 0),
            (-7.5, -6.75, 0),
        )

        divider_c2 = DashedLine(
            (7.5, 6.75, 0),
            (7.5, -6.75, 0),
        )
        
        self.play(
            Create(divider_c1),
            Create(divider_c2),
            Create(divider_r1),
            Create(divider_r2),
            run_time=4,
        )
        self.wait(6)

        """
        20 | -n 96,102 | voice: 22 sec, anim: 23 sec
        When crossing over, the genes in both genomes with the same innovation numbers are lined up. 
        These genes are called matching genes. Genes that do not match are either disjoint or excess, 
        depending on whether they occur within or outside the range of the other parents innovation numbers. 
        """
        
        crossover_grid_font_scaling = 0.32  # mit font_scaling = 0.5 ist vis_box_width ~ 1.32 
        crossover_conn_genes_vis = {
            "cell2": VGroup(*cell2.nucleus.get_visual_genotype(font_scaling=crossover_grid_font_scaling)["connection_genes"]),
            "cell3": VGroup(*cell3.nucleus.get_visual_genotype(font_scaling=crossover_grid_font_scaling)["connection_genes"])
            }
        for cell, genes_vis in crossover_conn_genes_vis.items():
            crossover_conn_genes_vis[cell] = sorted(genes_vis, key=lambda x: int(x[0][-1].text[5:])) # Sort by innovation number (last char of text
        
        max_box_width = max([box[1].get_width() for box in crossover_conn_genes_vis["cell2"] + (crossover_conn_genes_vis["cell3"])])
        crossover_grid_padding = (13.5 - 8*max_box_width)/7  # ~0.41
        # (8*1.33 [box_width] + 7*0.42 [padding] + 2*0.75 [h_margin] = 13.5 [=required width & available width])
        
        crossover_conn_genes_vis["cell2"][0].move_to(((-7.5 + 0.75 + max_box_width/2 + 0*(crossover_grid_padding + max_box_width), 4.5, 0)))
        crossover_conn_genes_vis["cell2"][1].move_to(((-7.5 + 0.75 + max_box_width/2 + 1*(crossover_grid_padding + max_box_width), 4.5, 0)))
        crossover_conn_genes_vis["cell2"][2].move_to(((-7.5 + 0.75 + max_box_width/2 + 2*(crossover_grid_padding + max_box_width), 4.5, 0)))
        crossover_conn_genes_vis["cell2"][3].move_to(((-7.5 + 0.75 + max_box_width/2 + 4*(crossover_grid_padding + max_box_width), 4.5, 0)))
        crossover_conn_genes_vis["cell2"][4].move_to(((-7.5 + 0.75 + max_box_width/2 + 5*(crossover_grid_padding + max_box_width), 4.5, 0)))
        crossover_conn_genes_vis["cell2"][5].move_to(((-7.5 + 0.75 + max_box_width/2 + 6*(crossover_grid_padding + max_box_width), 4.5, 0)))

        crossover_conn_genes_vis["cell3"][0].move_to(((-7.5 + 0.75 + max_box_width/2 + 0*(crossover_grid_padding + max_box_width), 0, 0)))
        crossover_conn_genes_vis["cell3"][1].move_to(((-7.5 + 0.75 + max_box_width/2 + 1*(crossover_grid_padding + max_box_width), 0, 0)))
        crossover_conn_genes_vis["cell3"][2].move_to(((-7.5 + 0.75 + max_box_width/2 + 2*(crossover_grid_padding + max_box_width), 0, 0)))
        crossover_conn_genes_vis["cell3"][3].move_to(((-7.5 + 0.75 + max_box_width/2 + 3*(crossover_grid_padding + max_box_width), 0, 0)))
        crossover_conn_genes_vis["cell3"][4].move_to(((-7.5 + 0.75 + max_box_width/2 + 5*(crossover_grid_padding + max_box_width), 0, 0)))
        crossover_conn_genes_vis["cell3"][5].move_to(((-7.5 + 0.75 + max_box_width/2 + 7*(crossover_grid_padding + max_box_width), 0, 0)))

        self.wait(6)
        self.play(Succession(  # Display matching genes
            AnimationGroup(
                FadeIn(crossover_conn_genes_vis["cell2"][0]),
                FadeIn(crossover_conn_genes_vis["cell3"][0]),
            ),
            AnimationGroup(
                FadeIn(crossover_conn_genes_vis["cell2"][1]),
                FadeIn(crossover_conn_genes_vis["cell3"][1]),
            ),
            AnimationGroup(
                FadeIn(crossover_conn_genes_vis["cell2"][2]),
                FadeIn(crossover_conn_genes_vis["cell3"][2]),
            ),
            AnimationGroup(
                FadeIn(crossover_conn_genes_vis["cell2"][4]),
                FadeIn(crossover_conn_genes_vis["cell3"][4]),
            ),
            run_time=4
        ))
        
        self.wait(2)
        self.play( # Display disjoint genes
            FadeIn(crossover_conn_genes_vis["cell2"][3]),
            FadeIn(crossover_conn_genes_vis["cell2"][5]),
            FadeIn(crossover_conn_genes_vis["cell3"][3]),
            run_time=1
        )
        self.wait(1)
        self.play( # Display excess genes
            FadeIn(crossover_conn_genes_vis["cell3"][5]),
            run_time=1
        )
        self.wait(9)

        """
        21 | -n 103,115 | voice: 4 sec, anim: 9 sec
        Matching genes are inherited randomly from either parent.
        """

        # Create child (genes are taken 'randomly' from cell2 and cell3 - {3 and 11 from cell3, rest from cell2 due to higher fitness})
        cell4_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [3, 10, 11, 12],
            "in_node": [6, 8, 4, 1],
            "out_node": [10, 10, 10, 9],
            "weight": [2.0, 0.3, 1.4, 2.4],
            "is_disabled": [False, False, False, False],
        })])

        cell4 = Cell(width=3, height=3, genes=(basis_node_genes, cell4_conn_genes), fitness=-150, num_waves=7, wobble_frequency=7)
        cell4.move_to((9.75, -4.5, 0))
        cell4_conn_genes_vis = Group(*cell2.nucleus.get_visual_genotype(font_scaling=crossover_grid_font_scaling)["connection_genes"])
        cell4_conn_genes_vis[0].move_to(((-7.5 + 0.75 + max_box_width/2 + 0*(crossover_grid_padding + max_box_width), -4.5, 0)))
        cell4_conn_genes_vis[1].move_to(((-7.5 + 0.75 + max_box_width/2 + 1*(crossover_grid_padding + max_box_width), -4.5, 0)))
        cell4_conn_genes_vis[2].move_to(((-7.5 + 0.75 + max_box_width/2 + 2*(crossover_grid_padding + max_box_width), -4.5, 0)))
        cell4_conn_genes_vis[3].move_to(((-7.5 + 0.75 + max_box_width/2 + 4*(crossover_grid_padding + max_box_width), -4.5, 0)))
        cell4_conn_genes_vis[4].move_to(((-7.5 + 0.75 + max_box_width/2 + 5*(crossover_grid_padding + max_box_width), -4.5, 0)))
        cell4_conn_genes_vis[5].move_to(((-7.5 + 0.75 + max_box_width/2 + 6*(crossover_grid_padding + max_box_width), -4.5, 0)))

        dominant_parent = ["cell2", "cell3", "cell3", "cell2", "cell3", "cell2"] # 'random' choices for crossover
        for idx, cell4_conn_gene_vis in enumerate(cell4_conn_genes_vis): # 4x1.8+2 secs, 4x3+1 anims
            if idx in [3, 5]:
                continue 
            self.play(
                crossover_conn_genes_vis[dominant_parent[idx]][idx][1].animate.set_stroke(color=colors["highlight"], width=2),
                run_time=0.7
            )
            self.play(
                FadeIn(cell4_conn_gene_vis),
                run_time=0.8
            )
            self.play(
                crossover_conn_genes_vis[dominant_parent[idx]][idx][1].animate.set_stroke(color=WHITE, width=1),
                run_time=0.3
            )

        self.wait(1.8)

        """
        22 | -n 116,123 | voice: 15 sec, anim: 17 sec
        Disjoint genes and excess genes on the other hand are inherited from the fittest parent.
        In our case the first parent has the higher fitness value, which is why all of his excess and disjoint genes are passed on to its offspring.
        """
        
        self.wait(10)
        for idx, cell4_conn_gene_vis in enumerate(cell4_conn_genes_vis): #  2 iterations a 1.8 sec
            if idx not in [3, 5]:
                continue 
            self.play(
                crossover_conn_genes_vis[dominant_parent[idx]][idx][1].animate.set_stroke(color=colors["highlight"], width=2),
                run_time=0.5
            )
            self.play(
                FadeIn(cell4_conn_gene_vis),
                run_time=0.8
            )
            self.play(
                crossover_conn_genes_vis[dominant_parent[idx]][idx][1].animate.set_stroke(color=WHITE, width=1),
                run_time=0.5
            )

        self.wait(3.4)

        """
        23 | -n 124,125 | voice: 8 sec, anim: 9 sec
        And thus, new life is born. Carrying genes from both parents, thereby enhancing the genetic diversity of the population.
        """

        self.play(
            FadeIn(cell4),
            run_time=2
        )
        self.wait(7)

        """ 
        24 | -n 126,126 | voice: 7 sec, anim: 8 sec
        With our foundational understanding of the NEAT algorithm in place, it's now time to put it to the test.        
        """

        # Blend to black in post productions
        self.wait(10)


class SceneFullSimulation(Scene00Setup):
    # Second half of the video where an actual implementation of the NEAT algorithm is visualized
    def construct(self):
        super().construct()

        self.camera.frame.scale_to_fit_height(9)

        # Create Map
        map = Rectangle(height=7.5, width=10.5, color=WHITE).move_to((-2.75, -0.75, 0))

        # Game Snipped
        game = Rectangle(height=3.375, width=5.5, color=WHITE).move_to((5.25, -2.8125, 0))

        # Create cell closeup
        closeup_box = Rectangle(height=5.625, width=5.5, color=WHITE).move_to((5.25, 1.6875, 0)).set_z_index(1)

        class Headline():
            def __init__(self):
                # Create headline box, which will contain text indicating current gen and phase
                self.box = Rectangle(height=1.5, width=10.5, color=WHITE).move_to((-2.75, 3.75, 0))
                self.old = None
                self.new = None

            def change(self, gen, phase):
                self.old = self.new
                self.new = (
                    CText(f"Generation {gen:003}", weight=BOLD).scale(0.7),
                    CText(phase).scale(0.7)
                )
                self.new[0].move_to((-8, 3.75, 0) + RIGHT*(self.new[0].width/2 + 0.4))
                self.new[1].move_to((2.5, 3.75, 0) + LEFT*(self.new[1].width/2 + 0.4))


        headline = Headline()

        self.add(game, map, closeup_box, headline.box)

        species_shapes = {  # Different shapes to distinguish different species
            0: Square().round_corners(radius=1).scale(0.9),
            1: Square().round_corners(radius=0.5).scale(0.9),
            2: Triangle().round_corners(radius=0.3).scale(1.3), # scaling to make all shapes the same size
            3: RegularPolygon(n=6).round_corners(radius=0.4),
            4: RegularPolygon(n=5).round_corners(radius=0.3),
        }
        # Due to lack of shapes, just pretend that the minorities belong to species 0
        for i in range(5, 10):
            species_shapes[i] = species_shapes[0].copy()
        # Set style for all shapes and scale them down evenly
        [shape.set_style(
            fill_color=colors["white"], 
            fill_opacity=1, 
            stroke_color=self.camera.background_color, 
            stroke_width=4,
            background_stroke_width=10,
            background_stroke_opacity=0,
            background_stroke_color=self.camera.background_color,
            ).scale(0.15) for shape in species_shapes.values()]

    
        # # ========================= Debugging (delete for final render)
        # non_parent_sample = 5  # has to be bigger than pop_size * elimination_rate
        # # ========================= Debugging

        def get_data(gen):
            """
            This function is used to obtain the data necessary to visualize a given generation. 

            Notes on which file contains which data:
            1. Spawning: crossover.offspring_before_mutation von t-1 
            2. Mutation: species.genotype von t0 (oder crossover.offspring von t-1)
            3. Species: species von t0 
            4. Fitness: fitness_perspecies von t0
            5. Elimination: species von t0 mit crossover.parents von t0 abgleichen
            6. Crossover: crossover.parents von t0
            """

            t0_cells = {}
            
            # Create a dict containing the uuid and species for all genotypes after mutation
            species_file_t0 = read_file(generation=gen, run=run, file="species")
            for i, species in enumerate(species_file_t0):
                for genotype in species.genotypes: 
                    t0_cells[genotype.uuid] = {"species": i, "genotype_after_mutation": genotype}

            # t0_cells.pop(list(t0_cells.keys())[0]) # important! remove first genotype, because it doesnt exist in crossings_tm1 (unclear why)

            # Check if uuid will be parent or eliminated
            crossing_file_t0 = read_file(generation=gen, run=run, file="20240104_225458_crossings_new")
            unique_parents_uuids_t0 = set([crossing[parent].uuid for crossing in crossing_file_t0.values() for parent in ["parent1", "parent2"]])
            for uuid in t0_cells.keys():
                t0_cells[uuid]["is_parent"] = uuid in unique_parents_uuids_t0
                # t0_cells[uuid]["children_uuids"] = [crossing["offspring"].uuid for crossing in crossing_file_t0.values() if uuid in [crossing["parent1"].uuid, crossing["parent2"].uuid]]

            # # ========================= Debugging (delete for final render)
            # # cap t0_cells for faster rendering during development. Keep all parents and a few non_parents
            # t0_parents = {uuid:genotype for uuid,genotype in t0_cells.items() if genotype["is_parent"]}
            # t0_non_parents = {uuid:genotype for uuid,genotype in t0_cells.items() if not genotype["is_parent"]}
            # t0_non_parents_sample = {uuid:t0_non_parents[uuid] for i, uuid in enumerate(t0_non_parents.keys()) if i < non_parent_sample}
            # t0_cells = t0_parents | t0_non_parents_sample
            # # ========================= Debugging

            # Get fitness of each genotype
            fitness_file_t0 = read_file(generation=gen, run=run, file="fitness_perspecies")
            fitness_list = {fitness_file_t0[species][1][i].uuid: fitness_file_t0[species][0][i] for species in fitness_file_t0 for i in range(len(fitness_file_t0[species][1]))}
            for uuid in t0_cells.keys():
                t0_cells[uuid]["fitness"] = fitness_list[uuid]

            # Get the genotype before mutation
            if gen != 0:
                crossing_file_tm1 = read_file(generation=gen-1, run=run, file="20240104_225458_crossings_new")    
                crossing_file_tm1_restructured = {crossing["offspring_before_mutation"].uuid: crossing for crossing in crossing_file_tm1.values()}            
                for uuid in t0_cells.keys():
                    t0_cells[uuid]["genotype_before_mutation"] = crossing_file_tm1_restructured[uuid]["offspring_before_mutation"]
                    t0_cells[uuid]["parents"] = (crossing_file_tm1_restructured[uuid]["parent1"].uuid, crossing_file_tm1_restructured[uuid]["parent2"].uuid)
                
                species_file_tm1 = read_file(generation=gen-1, run=run, file="species")
                species_per_genotype_uuid = {genotype.uuid: i for i, species in enumerate(species_file_tm1) for genotype in species.genotypes}
                for uuid in t0_cells.keys():
                    t0_cells[uuid]["parent_species"] = species_per_genotype_uuid[t0_cells[uuid]["parents"][0]] 

            # Cell Creation
            def create_cell():
                cell = species_shapes[t0_cells[uuid]["parent_species"] if gen != 0 else 0].copy()
                cell.move_to((
                    random.uniform(-8 + cell.get_width() + 0.075, 2.5 - cell.get_width() - 0.075), # + 0.075 for additional padding
                    random.uniform(-4.5 + cell.get_height() + 0.075, 3 - cell.get_height() - 0.075), 
                    0
                    )).set_z_index(random.random()).scale(random.uniform(0.9, 1.1))
                
                return cell


            for i, uuid in enumerate(t0_cells.keys()):  
                t0_cells[uuid]["cell"] = create_cell()

            for crossing in crossing_file_t0.values():
                crossing["offspring_cell"] = create_cell()

            return t0_cells, crossing_file_t0


        def animate_generation(
                gen, pop_new, crossovers_new, show_closeup=None, crossovers_old=None, pop_old=None, pause_after_clip=0, clip_repetitions=1,
                secs_per_phase={"Transition": 6, "Mutation": 5.5, "Speciation": 5.5, "Fitness Evaluation": 3.5, "Elimination": 8.5, "Crossover": 16.5}
                ):
            """
            This function contains the animations for a single generation. It will be executed 8 times in total, once for each generation.

            Attributes:
                gen (int): The generation number
                pop_new (dict): A dictionary containing the genotypes of the current generation
                crossovers_new (dict): A dictionary containing the crossover information of the current generation
                show_closeup (bool): If True, a closeup of the fittest individual will be shown
                crossovers_old (dict): A dictionary containing the crossover information of the previous generation
                pop_old (dict): A dictionary containing the genotypes of the previous generation (deprecated)
                pause_after_clip (int): The number of seconds to pause after the clip of the lunar lander has been shown
                clip_repetitions (int): The number of times the lunar lander clip will be repeated
                secs_per_phase (dict): A dictionary containing the duration of each phase in seconds.
            """

            # == Manual Adjustments for better storytelling == 
            if gen == 11:
                pop_new["20240104225625232840_4b79f080-b7b8-45f8-b254-f709c3224fe1"]["fitness"] = -111
            
            # ============== Transition between Generations ==============
            phase_total_time = 12 if show_closeup else 6
            rtm = secs_per_phase["Transition"]/phase_total_time  # rtm = run_time_multiplier

            print(f"== Generation {gen:003} - Transition")
            
            if gen == 0: # Grow cells if gen == 0
                headline.change(gen, "Initialisation")
                self.play(
                    Write(headline.new[0]), Write(headline.new[1]), 
                    run_time=1*rtm 
                    )

                self.play(LaggedStart(
                *[GrowFromCenter(pop_new[uuid]["cell"], run_time=2) for uuid in pop_new.keys()],
                lag_ratio=calc_lag_ratio(10, 2, len(pop_new)),
                run_time=5*rtm
            ))

            # Copy cell positions if two gens are consecutive
            elif all([(indv in [crossover["offspring"].uuid for crossover in crossovers_old.values()]) for indv in pop_new]):
                headline.change(gen, "Transition")
                self.play(
                    ReplacementTransform(headline.old[0], headline.new[0]), ReplacementTransform(headline.old[1], headline.new[1]),
                    run_time=1*rtm
                    )
                
                relevant_old_cells = {crossover["offspring"].uuid: crossover["offspring_cell"] for crossover in crossovers_old.values()}
                morph_animations = []
                for uuid, indv in pop_new.items():  # adjust position of t0 cells to match tm1 cells
                    indv["cell"].move_to(relevant_old_cells[uuid].get_center())
                    # indv["cell"].set_z_index(relevant_old_cells[uuid].get_z_index())
                    morph_animations.append(ReplacementTransform(relevant_old_cells[uuid], indv["cell"]))
                self.play(*morph_animations, run_time=5*rtm)
                # self.remove(*[offspring_cell for offspring_cell in relevant_old_cells.values()])
                # self.add(*[indv["cell"] for indv in pop_new.values()])
                # self.wait(5*rtm)

            else: # Morph transition if generations are non-consecutive 
                headline.change(gen, "Transition")
                self.play(
                    ReplacementTransform(headline.old[0], headline.new[0]), ReplacementTransform(headline.old[1], headline.new[1]), 
                    run_time=1*rtm
                )

                self.play(Succession(
                    AnimationGroup(*[FadeOut(crossover["offspring_cell"]) for crossover in crossovers_old.values()]),
                    AnimationGroup(*[FadeIn(indv["cell"]) for indv in pop_new.values()]),
                    run_time=5*rtm
                ))

            # === Initialize Closeup === 
            if show_closeup:
                closeup = Closeup(run=run, gen=gen)
                best_uuid = closeup.fittest_indv_uuid
                zoom_square = Square(side_length=0.6, color=colors["highlight"]).move_to(pop_new[best_uuid]["cell"].get_center())
                closeup_box.set_z_index(1)

                self.play(
                    FadeIn(zoom_square),
                    closeup_box.animate.set_stroke(color=colors["highlight"]),
                    FadeIn(closeup.membrane),
                    FadeIn(closeup.nucleus_before_mutation),
                    FadeIn(closeup.labels),
                    run_time=4*rtm
                )
                self.play(
                    closeup_box.animate.set_stroke(color=colors["white"]),
                    run_time=2*rtm
                )
                closeup_box.set_z_index(0)

            # ============== Mutation ==============
            phase_total_time = 5.5
            rtm = secs_per_phase["Mutation"]/phase_total_time 

            print(f"== Generation {gen:003} - Mutation")
            headline.change(gen, "Mutation")
            self.play(FadeOut(headline.old[0], headline.old[1]), FadeIn(headline.new[0], headline.new[1]), run_time=0.5*rtm)

            mutation_animations = []
            for uuid in pop_new:
                mutation_subanimations = [
                    [AnimationGroup(pop_new[uuid]["cell"].animate.set_background_stroke(color=colors["mutation"], opacity=1))], 
                    [AnimationGroup(pop_new[uuid]["cell"].animate.set_background_stroke(opacity=0))]
                    ]

                if show_closeup and uuid == best_uuid:
                    mutation_subanimations[0].append(AnimationGroup(ColorChangeAnimation(closeup.membrane, colors["mutation"], closeup.wobble)))
                    mutation_subanimations[1].append(AnimationGroup(ColorChangeAnimation(closeup.membrane, colors["white"], closeup.wobble)))
                    mutation_subanimations.insert(1, [AnimationGroup(ReplacementTransform(closeup.nucleus_before_mutation, closeup.nucleus_after_mutation, run_time=2))])

                mutation_animations.append(Succession(
                    *[AnimationGroup(*subanimation) for subanimation in mutation_subanimations]
                )) 

            self.play(LaggedStart(
                *mutation_animations,
                lag_ratio=calc_lag_ratio(5*rtm, 1, len(pop_new)),
                run_time=5*rtm,
            ))

            # ============== Speciation ==============
            phase_total_time = 5.5
            rtm = secs_per_phase["Speciation"]/phase_total_time 

            print(f"== Generation {gen:003} - Speciation")
            headline.change(gen, "Speciation")
            self.play(FadeOut(headline.old[0], headline.old[1]), FadeIn(headline.new[0], headline.new[1]), run_time=0.5*rtm)

            if gen != 0:
                affected_uuids = [uuid for uuid in pop_new.keys() if pop_new[uuid]["parent_species"] != pop_new[uuid]["species"]]
                new_cells = [species_shapes[pop_new[uuid]["species"]].copy().scale(random.uniform(0.9, 1.1)).move_to(pop_new[uuid]["cell"].get_center()).set_background_stroke(color=colors["speciation"], opacity=1) for uuid in affected_uuids]
                        
                self.play(
                    AnimationGroup(*[pop_new[uuid]["cell"].animate.set_background_stroke(color=colors["speciation"], opacity=1) for uuid in affected_uuids], run_time=2*rtm)
                )
                self.play(
                    AnimationGroup(*[ReplacementTransform(pop_new[uuid]["cell"], new_cell) for uuid, new_cell in zip(affected_uuids, new_cells)], run_time=2*rtm)
                )
                self.play(
                    AnimationGroup(*[new_cell.animate.set_background_stroke(color=self.camera.background_color, opacity=0) for new_cell in new_cells], run_time=1*rtm)
                )

                for i, uuid in enumerate(affected_uuids):
                    pop_new[uuid]["cell"] = new_cells[i]  # dict manipulation
                        
            if gen == 0 or affected_uuids.__len__() == 0:
                self.wait(5.5*rtm)

            # ============== Fitness Evaluation ==============
            phase_total_time = 3.5 # Cant be determined for show_closeup=True due to differing clip length
            rtm = secs_per_phase["Fitness Evaluation"]/phase_total_time 

            print(f"== Generation {gen:003} - Fitness Evaluation")
            headline.change(gen, "Fitness Evaluation")
            self.play(FadeOut(headline.old[0], headline.old[1]), FadeIn(headline.new[0], headline.new[1]), run_time=0.5*rtm)

            # fitnesses = [indv["fitness"] for indv in pop_new.values()]
            fitness_range = (-278, 278) # 278 best fitness overall

            fitness_animations = []
            for uuid in pop_new.keys():
                relative_fitness_color = ManimColor(RED).interpolate(GREEN, (pop_new[uuid]["fitness"] - fitness_range[0]) / (fitness_range[1] - fitness_range[0]))
                fitness_animations.append(pop_new[uuid]["cell"].animate.set_fill(color=relative_fitness_color))
                
            self.play(
                *fitness_animations,
                run_time=3*rtm,
                )
            
            # Lunar Lander Clip
            if show_closeup:
                game.set_z_index(1)
                best_species = pop_new[best_uuid]["species"]
                self.play(game.animate.set_stroke(color=colors["highlight"]), run_time=2)
                for i in range(clip_repetitions):
                    duration_in_frames = closeup.launch_lunar_lander(best_species)
                    self.wait(duration_in_frames / 50) # 50 fps - get secs
                self.play(game.animate.set_stroke(color=colors["white"]), run_time=2)
                game.set_z_index(0)

                self.wait(pause_after_clip)

            # ============== Elimination ==============
            phase_total_time = 8.5
            rtm = secs_per_phase["Elimination"]/phase_total_time 

            print(f"== Generation {gen:003} - Elimination")
            headline.change(gen, "Elimination")
            self.play(FadeOut(headline.old[0], headline.old[1]), FadeIn(headline.new[0], headline.new[1]), run_time=0.5)

            elimination_animations = []
            for genotype in pop_new.values():
                if not genotype["is_parent"]:
                    elimination_animations.append(Succession(
                        AnimationGroup(genotype["cell"].animate.set_background_stroke(color=colors["elimination"], opacity=1), run_time=2),
                        AnimationGroup(FadeOut(genotype["cell"], run_time=2*rtm)),
                    ))

            self.play(LaggedStart(
                *elimination_animations,
                lag_ratio=calc_lag_ratio(8*rtm, 4, len(pop_new)),
                run_time=8*rtm,
            ))

            # ============== Crossover ==============
            phase_total_time = 16.5
            rtm = secs_per_phase["Crossover"]/phase_total_time 

            print(f"== Generation {gen:003} - Crossover")
            headline.change(gen, "Crossover")
            self.play(FadeOut(headline.old[0], headline.old[1]), FadeIn(headline.new[0], headline.new[1]), run_time=0.5)

            crossover_animations = []
            for crossover in crossovers_new.values():
                if crossover["parent1"].uuid in pop_new and crossover["parent2"].uuid in pop_new: # debug
                    # If closeup enabled color in membrane when the corresponding cell is involved in a crossover
                    crossover_subanimations = [
                        [
                            AnimationGroup(pop_new[crossover["parent1"].uuid]["cell"].animate.set_background_stroke(color=colors["crossover"], opacity=1)),  
                            AnimationGroup(pop_new[crossover["parent2"].uuid]["cell"].animate.set_background_stroke(color=colors["crossover"], opacity=1)),
                            AnimationGroup(GrowFromCenter(crossover["offspring_cell"])),
                        ],
                        [
                            AnimationGroup(pop_new[crossover["parent1"].uuid]["cell"].animate.set_background_stroke(opacity=0)),  
                            AnimationGroup(pop_new[crossover["parent2"].uuid]["cell"].animate.set_background_stroke(opacity=0)),
                        ]
                    ]

                    if show_closeup and (crossover["parent1"].uuid == best_uuid or crossover["parent2"].uuid == best_uuid):
                        crossover_subanimations[0].append(AnimationGroup(ColorChangeAnimation(closeup.membrane, colors["crossover"], closeup.wobble)))
                        crossover_subanimations[1].append(AnimationGroup(ColorChangeAnimation(closeup.membrane, colors["white"], closeup.wobble)))

                    crossover_animations.append(Succession(
                        AnimationGroup(*crossover_subanimations[0]), AnimationGroup(*crossover_subanimations[1]), run_time=4
                    )) 
    
            self.play(LaggedStart(
                *crossover_animations,
                lag_ratio=calc_lag_ratio(8*rtm, 4, len(pop_new)),
                run_time=8*rtm
            ))
            
            # Parents perish
            parents = [indv for indv in pop_new.values() if indv["is_parent"]]
            perish_animations = []
            for parent in parents:
                perish_subanimations = [
                    [AnimationGroup(parent["cell"].animate.set_background_stroke(color=colors["elimination"], opacity=1))],
                    [AnimationGroup(FadeOut(parent["cell"]))]
                ]
                if show_closeup and parent["genotype_after_mutation"].uuid == best_uuid:
                    perish_subanimations[0].append(AnimationGroup(ColorChangeAnimation(closeup.membrane, colors["elimination"], closeup.wobble)))
                    perish_subanimations[1].extend([
                        AnimationGroup(FadeOut(closeup.membrane), FadeOut(closeup.labels), FadeOut(closeup.nucleus_after_mutation), FadeOut(zoom_square))
                        ])
                
                perish_animations.append(Succession(
                    *perish_subanimations[0], *perish_subanimations[1], run_time=4
                )) 

            self.play(LaggedStart(
                *perish_animations,
                lag_ratio=calc_lag_ratio(8*rtm, 4, len(pop_new)),
                run_time=8*rtm,
            ))


        """========== Simulation ============
        25 | -n , | voice: 7 sec, anim: 9 sec
        Let's initiate an actual simulation to observe its effectiveness in training neural networks to play Lunar Lander.
        """

        self.wait(9)

        """ ========= Gen 0 ==================
        26 | -n 0,13
        a. Transition
        First: Lets create an initial population of size 150. Each of these little dots represents a neural network like the ones we've seen before.

        b. Mutation: 
        Whenever one of them glows in blue, it indicates a mutation has occurred.

        c. Speciation:
        At this point the differences in their genome are still rather small, so they all belong to the same species.

        d. Fitness Evaluation:
        Initially set with random weights and after just one mutation, their fitness levels are quite low, ranging from minus 1600 to minus 85. 
        To put it in perspective, a score of at least 200 is required to solve the lunar lander problem. 
        For our purposes the fitness levels are represented by colors: red denotes a lower score, while green indicates a higher one.

        e. Elimination: 
        We will now eliminate the weakest 80% leaving us with the fittest 30 individuals.

        f. Crossover:
        The surviving individuals then reproduce to form the next generation, collectively creating 150 offspring.
        After that, the parents perish, setting the stage for the next generation.

        """

        gen0, crossovers0 = get_data(gen=0)
        animate_generation(gen=0, pop_new=gen0, crossovers_new=crossovers0, show_closeup=False,
                           secs_per_phase={"Transition": 12, "Mutation": 6, "Speciation": 6, "Fitness Evaluation": 28, "Elimination": 7, "Crossover": 18}
                           )


        """ ========= Gen 1 ==================
        27 | -n 14,42
        a. Transition
        We are now inside the second generation, meaning one iteration of the NEAT algorithm has passed and the networks should be a tiny bit better at steering the aircraft.
        Lets test this hypothesis, by looking at a random individual of the current population throughout this generation.

        b. Mutation: 
        Save for some minor changes to the weights of its connection genes the individuals genome has remained unchanged from mutation.

        c. Speciation:
        Yet again there is not enough genetic diversity to warrant the creation of a new species.

        d. Fitness Evaluation:
        The fitness levels remain very low. There is even a slight decrease in the average fitness.
        But what does that mean for the individuals ability to solve the lunar lander problem?
        Let's watch and find out!

        d2. Verdict:
        As you may have already expected, the aircraft plummets uncontrollably from the sky, resulting in an unmitigated crash on the moon's surface.
        Remember: Node 0 is just a bias node, so until there is a mutation connecting its input to its output nodes, the network has no means of 
        using information about its environment to guide its actions. 

        e. Elimination:
        Yet again, the less fortunate 80% of the population succumbs to the forces of natural selection.

        f. Crossover:
        And once more new life emerges as the remaining individuals reproduce, thereby paving the way for the next generation.
        """

        gen1, crossovers1 = get_data(gen=1)
        animate_generation(
            gen=1, pop_new=gen1, pop_old=gen0, crossovers_old=crossovers0, crossovers_new=crossovers1, show_closeup=True, pause_after_clip=19, clip_repetitions=8,
            secs_per_phase={"Transition": 21, "Mutation": 11, "Speciation": 5, "Fitness Evaluation": 15, "Elimination": 8, "Crossover": 9}
        )
 

        """ ========= Gen 11 ==================
        28 | -n 43,70
        a. Transition
        Let us now fast forward to the 11th generation. Moving forward the close-up view will always feature the fittest individual of each generation.

        b. Mutation: 
        After the usual mutation step something interesting happens.

        c. Speciation:
        A new species emerges. Initially comprised of just two individuals.

        d. Fitness Evaluation:
        Let's see if there are any meaningful improvements regarding the networks performance.

        d2. Verdict:
        That doesn't seem to be the case.

        e. Elimination:
        To protect topological innovation, the NEAT algorithm ensures that new species are not eliminated prematurely.

        f. Crossover:
        Let us now jump to generation 34, where another milestone is archived. 
        
        """

        gen11, crossovers11 = get_data(gen=11)
        animate_generation(
            gen=11, pop_new=gen11, pop_old=gen1, crossovers_new=crossovers11, crossovers_old=crossovers1, show_closeup=True, pause_after_clip=3, clip_repetitions=8,
            secs_per_phase={"Transition": 11, "Mutation": 4, "Speciation": 9, "Fitness Evaluation": 6, "Elimination": 7, "Crossover": 9}
        )


        """ ========= Gen 34 ==================  
        29 | -n 71,98

        a. Initalization
        This generation is particularly interesting, because it is the first time that one of the individuals has managed to score a positive fitness value.

        b. Mutation & c. Speciation & d. Fitness Evaluation:
        By now the individual has a total of 5 connection genes. 
        Its strongest edge has a weight of 11.32 and connects the angle of the aircraft to its horizontal thrust.
        That indicates that the network might have learned to balance itself out by applying thrust in the opposite direction of its tilt.

        The remaining connections are not as strong, but seem to be quite useful nonetheless.
        One of them links the position on the x-axis to the horizontal thrust, effectively allowing the aircraft to steer left and right.

        Other connections are a bit more dubious, for example the one connecting the left leg of the aircraft to its horizontal thrust. 
        Luckily its weight is close to zero, so it doesn't have much of an impact on the aircrafts behavior.

        But enough with the talk - let us see it in action.
        
        d2. Verdict:
        While it's not as impressive as one might have hoped, it's still a significant improvement over the previous generations.
        The lander is actively softening its fall by applying downward thrust and tilting itself to the right in order to reach the designated landing pad.
        
        e. Elimination & f. Crossover:
        However it is still far from perfect, so let us now jump 67 generations and see how the algorithm has progressed.
                
        """

        gen34, crossovers34 = get_data(gen=34)
        animate_generation(
            gen=34, pop_new=gen34, pop_old=gen11, crossovers_new=crossovers34, crossovers_old=crossovers11, show_closeup=True, pause_after_clip=4, clip_repetitions=8,
            secs_per_phase={"Transition": 11, "Mutation": 19, "Speciation": 19, "Fitness Evaluation": 19, "Elimination": 5.5, "Crossover": 5.5}
        )


        """ ========= Gen 101 ================== 
        30 | -n 99, 132

        a. Initalization & b. Mutation & c. Speciation & d. Fitness Evaluation:
        As one can see straight away, the number of species has increased significantly.
        By now some of them have developed their own strategy at playing the game, competing with species 0 which has held the top spot for quite some time.
        Species 2 for example - recognizable by its triangular shape - shows a somewhat peculiar behavior.  

        d2. Verdict:
        It prioritizes a quick arrival over a safe landing, letting itself fall from the sky and only applying thrust at the very last moment.
        An approach that is rewarded with a score slightly above 0. However this behaviour seems to be more of a lucky coincidence than a well thought out strategy.
        Looking at its phenotype we can see that vertical thrust only gets applied when angular velocity increases. 
        In other words: The aircraft slows its fall whenever it tilts out of control.
        Since the tilt tends to increase throughout the fall, the aircraft just happens to apply downward thrust at the right moment to dampen its landing.

        e. Elimination & f. Crossover:
        While this approach seems to be working surprisingly well, we sure can do better!
        """

        gen101, crossovers101 = get_data(gen=101)
        animate_generation(
            gen=101, pop_new=gen101, pop_old=gen34, crossovers_new=crossovers101, crossovers_old=crossovers34, show_closeup=True, pause_after_clip=23, clip_repetitions=14,
            secs_per_phase={"Transition": 6.5, "Mutation": 5.75, "Speciation": 5.75, "Fitness Evaluation": 6.5, "Elimination": 4.5 , "Crossover": 4.5}
        ) # pause_after_clip = 42 (length of d2 voice over) - 1.62 (clip_length) * 12 (clip_repetitions - 2)


        """ ========= Gen 150 ================== 
        31 | -n 133, 153

        a. Initalization & b. Mutation & c. Speciation & d. Fitness Evaluation:
        In generation 150 species 1 reigns supreme, breaking the all time record with a fitness value of 70.

        d2. Verdict:
        It approaches the landing pad cautiously, carefully adjusting its position and velocity to ensure a safe landing.
        Unfortunately it doesn't quite understand that the job is done once it has touched the moons surface, so it keeps applying thrust.
        Since one leg keeps bouncing of the floor, the game doesn't register a successful landing and the time keeps ticking, negatively affecting the score.

        e. Elimination & f. Crossover:
        Over the course of the next generations Species 1 will continue to refine its strategy, ultimately achieving a fitness value of 100, and soon thereafter, surpassing it to reach 200.
        """

        gen150, crossovers150 = get_data(gen=150)
        animate_generation(
            gen=150, pop_new=gen150, pop_old=gen101, crossovers_new=crossovers150, crossovers_old=crossovers101, show_closeup=True, pause_after_clip=10, clip_repetitions=1,
            secs_per_phase={"Transition": 3, "Mutation": 3, "Speciation": 2, "Fitness Evaluation": 2, "Elimination": 6, "Crossover": 6}
        ) # done


        """ ========= Gen 178 ================== 
        32 | -n 154,179

        a. Initalization & b. Mutation & c. Speciation & d. Fitness Evaluation:
        That makes species 1 the first to officially complete the game. So let's not waste any time and look at how it plays the game.

        d2. Verdict:
        It maneuvers the aircraft with precision, ensuring a safe landing near the pad. 
        As can be seen here, ending on the landing pad is actually not strictly required, although it does help to achieve a higher score.
        Notably, the network applies thrust to the right side of the aircraft towards the end, likely to adjust its tilt. 
        However, this action inadvertently prevents the aircraft from sliding onto the pad.
        
        e. Elimination & f. Crossover:
        All in all the network is way better than the ones we've started with, but there's still room for growth. 

        """

        gen178, crossovers178 = get_data(gen=178)
        animate_generation(
            gen=178, pop_new=gen178, pop_old=gen150, crossovers_new=crossovers178, crossovers_old=crossovers150, show_closeup=True, pause_after_clip=2, clip_repetitions=6,
            secs_per_phase={"Transition": 2, "Mutation": 2, "Speciation": 2, "Fitness Evaluation": 1, "Elimination": 2, "Crossover": 4}
        ) # done


        """ ========= Gen 348 ================== 
        33 | -n 180

        a. Initalization & b. Mutation & c. Speciation & d. Fitness Evaluation:
        We will now look at one last generation - Generation 348 to be precise, as it contains the very best individual that the algorithm has produced in a total of 400 iterations.
        Right out of the gate, one can see that species 0 has displaced all others and that the fittest individual possesses two hidden layers, each containing one node. Let us now take a look at its connections, starting with the strongest one. 

        The edge connecting node 4 and 11, links the y-velocity to the second hidden node, which then ties to the vertical thrust. This setup allows the aircraft to decelerate its descent in proportion to its current velocity.
        The next strongest link connects the angle to the horizontal thrust. This is the same connection that species 0 had in generation 34, aiding the aircraft in maintaining balance.
        Interestingly, it only utilizes the y-velocity for this purpose, disregarding the y-coordinate entirely.
        The third key connection ties the x-velocity to the horizontal thrust, likely enabling controlled left and right steering. 
        There are also some connections whose purpose is not immediately clear, like the one linking the bias and right leg to the first hidden node, which in turn influences the horizontal thrust.
        
        Similar to genetic traits in nature, these connections might not be immediately beneficial, but as long as they don't present a selective disadvantage, they may persist.
        And now, for the last time. Let's launch the aircraft and witness the final result of the NEAT algorithm.

        d2. Verdict:
        The lunar lander gracefully approaches the moon's surface, its descent marked by meticulous precision and careful, fluid maneuvers. 
        Every action is deliberate, culminating in a gentle, controlled touchdown precisely between the two flags.
        
        In this journey, over 52,200 individuals emerged and faded away. 
        Eight distinct species engaged in a relentless struggle for supremacy in this virtual landscape.
        Innumerable mutations occured. Countless genese where crossed.
        Now, this final entity stands tall, representing the pinnacle of evolution.

        A total of 8 different species fought for suprimacy in this virtual landscape.
        Innumerable mutations occured - countless genese where crossed.

        Thank you for your attention.

        [FADE TO BLACK]

        """

        gen348, crossovers348 = get_data(gen=348)
        animate_generation(
            gen=348, pop_new=gen348, pop_old=gen178, crossovers_new=crossovers348, crossovers_old=crossovers178, show_closeup=True, pause_after_clip=10, clip_repetitions=5,
            secs_per_phase={"Transition": 30, "Mutation": 30, "Speciation": 13, "Fitness Evaluation": 18, "Elimination":1, "Crossover":1}
        )  # pause_after_clip = 30 (length of d2 voice over) - 6.22 (clip_length) * 1 2 (clip_repetitions - 2)


class SceneCloseUp(Scene00Setup):
    """
    This class is used to create the closeup of a cell while the lunar lander video is playing. It animates the activation of the cells neurons during the video.
    This could not be integrated into the SceneFullSimulation due to inherent rendering limitations. Has to be added to final video in post production.
    """
    def construct(self):
        super().construct()

        gen = 348
        reps  = 1
        best_species=0

        self.camera.frame.scale_to_fit_height(9)

        c = Closeup(gen=gen, run=run)
        self.add(c.membrane, c.nucleus_after_mutation, c.labels)

        for _ in range(reps):
            duration_in_frames = c.launch_lunar_lander(best_species=best_species)
            duration_in_secs = duration_in_frames / 50 # 50 fps
            self.wait(duration_in_secs)