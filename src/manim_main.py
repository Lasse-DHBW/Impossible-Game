import os
from manim import *
from manim_utils import read_file, decypher_genotype, calc_lag_ratio
from manim_mobjects import Cell, Nucleus
from manim_utils import CText
import pandas as pd
import random
import json


# Debugging
random.seed = 42

colors = {
    "highlight": YELLOW_D,  # Bei Änderung: Auch in nucleus LabeledDot anpassen
    "mutation": BLUE_D,
    "elimination": RED_D,
    "crossover": GREEN_D,
    "white": GRAY_A,
}

# Neat
# run = '2023_12_07_19_59_20_340927'

run = "20231220_115303"


class Scene00Setup(MovingCameraScene):
    def construct(self):
        self.camera.background_color = "#222222"
        self.camera.frame.scale_to_fit_height(16)
        config["frame_rate"] = 50

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
        cell = Cell(width=cell_height/2, height=cell_height, genes=genes, fitness=-500, num_waves=6, wobble_frequency=3)

        # Set opacity of nodes and edges to 0.5 - cant use nucleus.set_opacity because of LabeledDot.background_stroke_opacity
        cell.nucleus.set_fill(opacity=0.5)
        cell.nucleus.set_stroke(opacity=0.5)

        self.play(  #1
            FadeIn(cell),
            run_time=3
            )

        # Zoom in
        v_margin, h_margin = 1.5, 0.5
        self.play( #2
            self.camera.frame.animate.scale_to_fit_height(cell_height + v_margin*2), 

            run_time=1
            )

        """
        Its nodes represent neurons, similar to those in our brain, and the pathways are the synaptic connections. 
        Our organism's brain, if you will, comprises nine input nodes, mirroring sensory information from the 
        Lunar Lander game – the lander's position, velocity, angle, and contact sensors, plus an additional bias node, 
        always set to one.
        """

        # Highlight input nodes
        for vertice in cell.nucleus.vertices:
            if vertice in cell.nucleus.partitions[0]:
                cell.nucleus[vertice].set_background_stroke(opacity=1)
                cell.nucleus[vertice].set_fill(opacity=1)
                self.wait(0.2) # 3-11
                cell.nucleus[vertice].set_background_stroke(opacity=0)

        """
        But what about the outputs? This organism has to make decisions, after all. 
        Here, it boils down to two critical numbers: one controlling the main engine, another dictating the 
        lateral thrusters. These outputs are its actions, its means of interacting with its environment – 
        in this case, the challenging terrain of the lunar surface.
        """

        # Highlight output nodes
        for vertice in cell.nucleus.vertices:
            if vertice in cell.nucleus.partitions[1]:
                cell.nucleus[vertice].set_background_stroke(opacity=1)
                cell.nucleus[vertice].set_fill(opacity=1)
                self.wait(0.2) # 12-13
                cell.nucleus[vertice].set_background_stroke(opacity=0)

        """
        This most basic neural network there are only 2 connections, both connecting one of the outputs 
        with the bias node.  meaning that the inputs aren't influencing the actions of the lunar lander at all. 
        This is obviously a pretty poor architecture, since the gamestate is in no way contributing to the 
        actions that the lunar lander takes, meaning that information like the landers coordinates and velocity 
        are entirely disregarded.
        """

        # Highlight connections
        for edge in cell.nucleus.edges.values():
            edge.set_color(colors["highlight"])
            edge.set_stroke(opacity=1)
            self.wait(0.2) # 14-15
            edge.set_color(WHITE)
        
        self.wait(1)

        """
        This issue will soon be adressed through the process of mutation. But before we look into that, 
        we must understand how the current architecture of the network is expressed in the NEAT framework. 
        """

        # Move to side
        new_cell_pos = LEFT*(self.camera.frame_width / 2 - cell.width / 2 - h_margin) 
        self.play( #16
            cell.animate.move_to(new_cell_pos)
            )       

        # Add divider
        divider = DashedLine(
            self.camera.frame_height // 2 * UP + LEFT*(self.camera.frame_width / 2 - cell.width - h_margin*2), 
            self.camera.frame_height // 2 * DOWN + LEFT*(self.camera.frame_width / 2 - cell.width - h_margin*2),  
            dashed_ratio=0.3, 
            color=WHITE
            )
        
        self.play( #17
            Create(divider),
            cell.membrane.animate.set_opacity(0.5),
            cell.nucleus.edges[(0, 10)].animate.set_opacity(0.51),
            run_time = 1
            )

        self.wait(1) #18

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
            cell.membrane.animate.set_stroke(color=colors["mutation"]),
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
            cell.nucleus.edges[(0, 10)].animate.set_color(colors["highlight"]),
            weight.animate.set_color(colors["highlight"]),
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

        cell.nucleus.edges[(6, 10)].set_color(colors["highlight"]),
        new_conn_gene_vis[-1].set_stroke(color=colors["highlight"]),   

        self.play(
            GrowFromCenter(cell.nucleus.edges[(6, 10)]),
            GrowFromCenter(new_conn_gene_vis),
            run_time=2
            )

        self.play(
            cell.nucleus.edges[(6, 10)].animate.set_color(WHITE),
            new_conn_gene_vis[-1].animate.set_stroke(color=WHITE),
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

            run_time=3
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

        self.wait(2)

   
        """
        ========== Speciation ============
        Now that we understand how new genes are introduced into the genotype, lets zoom out
        and look at a small population of size 4, where each network was affected by different mutations.        
        """

        species_list = read_file(generation=0, run=run, file="species")
        basis_node_genes, basis_conn_genes = decypher_genotype(species_list[0].genotypes[0])

        cell0 = cell

        # Create genotype for cell 2
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

        # Create genotype for cell 3
        cell2_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [3, 10, 12, 11],
            "in_node": [6, 8, 1, 4],
            "out_node": [10, 10, 9, 10],
            "weight": [1.3, 0.3, 1.2, 2.4],
            "is_disabled": [False, False, False, False],
        })])

        # Create genotype for cell 4
        cell3_conn_genes = pd.concat([basis_conn_genes, pd.DataFrame({
            "innovation_number": [7, 11, 3, 14],
            "in_node": [2, 4, 6, 5],
            "out_node": [10, 10, 10, 9],
            "weight": [0.5, 1.4, 2.0, 0.9],
            "is_disabled": [False, False, False, False],
        })])

        # Zoom out to aspect ratio 13.5 : 24

        cell1 = Cell(width=cell_height/2, height=cell_height, genes=(cell1_node_genes, cell1_conn_genes), fitness=-481, num_waves=8, wobble_frequency=4)
        cell1.move_to((-0.5, 2.7, 0))
        cell2 = Cell(width=3, height=3, genes=(basis_node_genes, cell2_conn_genes), fitness=-283, num_waves=8, wobble_frequency=2)
        cell2.move_to((1.4, -2.5, 0))
        cell3 = Cell(width=3, height=3, genes=(basis_node_genes, cell3_conn_genes), fitness=-330, num_waves=6, wobble_frequency=4)
        cell3.move_to((5.9, -1.2, 0))
        self.add(cell1, cell2, cell3)

        angles = [PI*0.22, PI*0.23, -PI*0.31]
        [cell.rotate(angle) for cell, angle in zip([cell1, cell2, cell3], angles)]

        self.play(FadeOut(VGroup(*grid.values())),
                  run_time=2)

        self.play(
            self.camera.frame.animate.scale_to_fit_height(13.5),
            FadeIn(cell1),
            FadeIn(cell2),
            FadeIn(cell3),
            run_time=2
            )

        """
        ========== Fitness ============
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

            run_time=2
        )

        """
        ========== elimination ============
        """

        self.play(
            cell0.membrane.animate.set_stroke(color=colors["elimination"], width=2),
            cell1.membrane.animate.set_stroke(color=colors["elimination"], width=2),
            run_time=2,
        )

        self.play(
            FadeOut(cell0.membrane),
            FadeOut(cell0.fitness),
            FadeOut(cell1.membrane),
            FadeOut(cell1.fitness),
            run_time=2,
        )


        """
        ============= Cross Over ================
        """

        print(cell2.get_center(), cell3.get_center())

        self.play(
            FadeOut(cell2.fitness),
            FadeOut(cell3.fitness),
            FadeIn(cell2.nucleus),
            FadeIn(cell3.nucleus),
            run_time=1,
        )
        cell2.fitness.set_fill(opacity=0)
        cell3.fitness.set_fill(opacity=0)


        self.play(
            cell2.membrane.animate.set_stroke(color=colors["crossover"], width=2),
            cell3.membrane.animate.set_stroke(color=colors["crossover"], width=2),
        )


        # Cross Over Grid: 3x3 Grid,
        # Frame: 13.5 : 24
        # row height = 3 (cell_height) + 2*0.75 (v_margin) = 4.5
        # col width for cols 1 & 3 = 3 (cell_width) + 2*0.75 (h_margin) = 4.5
        # middle col width = 24 - 2*4.5 = 15
        
        v_margin, h_margin = 0.75, 0.75 


        # Create Grid for cross over explanation
        self.play(
            cell2.animate.move_to((-9.75, 4.5, 0)),
            cell3.animate.move_to((-9.75, 0, 0)),
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
            Create(divider_r2)
        )

        
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

        for cell in crossover_conn_genes_vis.values():
            for box in cell:
                self.play(
                    Create(box),
                    run_time=0.2
                    )


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
        for idx, cell4_conn_gene_vis in enumerate(cell4_conn_genes_vis):
            if idx == 3:
                self.wait(1) # Pause to explain why gene with innov 7 is not inherited

            self.play(
                crossover_conn_genes_vis[dominant_parent[idx]][idx][1].animate.set_stroke(color=colors["highlight"], width=2),
                run_time=0.2
            )
            self.play(
                FadeIn(cell4_conn_gene_vis),
                run_time=0.5
            )
            self.play(
                crossover_conn_genes_vis[dominant_parent[idx]][idx][1].animate.set_stroke(color=WHITE, width=1),
                run_time=0.3
            )

       
        self.wait(1)

        self.play(
            FadeIn(cell4)
        )

        # self.play(
        #     FadeOut(cell2),
        #     FadeOut(cell3),
        #     FadeOut(divider_c1),
        #     FadeOut(divider_c2),
        #     FadeOut(divider_r1),
        #     FadeOut(divider_r2),
        #     FadeOut(VGroup(*crossover_conn_genes_vis.values())),
        #     FadeOut(cell4_conn_genes_vis),
        #     FadeOut(cell4)
        # )

        self.wait(1)


class SceneFullSimulation(Scene00Setup):
    def construct(self):
        super().construct()

        self.camera.frame.scale_to_fit_height(9)

        # Create Map
        map = Rectangle(height=7.5, width=10.5, color=WHITE).move_to((-2.75, -0.75, 0))

        # Game Snipped
        game = Rectangle(height=3.375, width=5.5, color=WHITE).move_to((5.25, -2.8125, 0))

        # Create cell closeup
        closeup = Rectangle(height=5.625, width=5.5, color=WHITE).move_to((5.25, 1.6875, 0))

        # Create headline box, which will contain text indicating current gen and phase
        headline_box = Rectangle(height=1.5, width=10.5, color=WHITE).move_to((-2.75, 3.75, 0))
        
        self.add(game, map, closeup, headline_box)

        def change_headline(gen, phase, headline_texts=None):
            if headline_texts is not None:
                self.remove(*headline_texts)

            gen_counter_text = CText(f"Generation {gen:003}", weight=BOLD).scale(0.7)
            gen_counter_text.move_to((-8, 3.75, 0) + RIGHT*(gen_counter_text.width/2 + 0.4))

            phase_ticker_text = CText(phase).scale(0.7)
            phase_ticker_text.move_to((2.5, 3.75, 0) + LEFT*(phase_ticker_text.width/2 + 0.4))

            return gen_counter_text, phase_ticker_text


        headline_texts = change_headline(0, "Initialisation")
        self.add(*headline_texts)

        species_shapes = {  # Different shapes to distinguish different species
            0: Square().round_corners(radius=1).scale(0.9),
            1: Square().round_corners(radius=0.5).scale(0.9),
            2: Triangle().round_corners(radius=0.3).scale(1.3), # scaling to make all shapes the same size
            3: RegularPolygon(n=6).round_corners(radius=0.4),
            4: RegularPolygon(n=5).round_corners(radius=0.3),
        }
        # Set style for all shapes and scale them down evenly
        [shape.set_style(
            fill_color=colors["white"], 
            fill_opacity=1, 
            stroke_color=self.camera.background_color, 
            stroke_width=4,
            background_stroke_width=10,
            background_stroke_color=self.camera.background_color,
            ).scale(0.15) for shape in species_shapes.values()]


        
        
        tm1_cells = {}  # Will contain parent cells of last gen

        # ========================= Debugging (delete for final render)
        debug_iterations = [0, 1]
        debug_sample = 35  # has to be bigger than pop_size * elimination_rate
        # ========================= Debugging

        for gen in range(131):  # There are 131 Generations
            # ========================= Debugging (delete for final render)
            if gen < debug_iterations[0] or gen > debug_iterations[1]:
                continue
            # ========================= Debugging

            # ==== Data Fetching ====
            # 1. Spawning: crossover.offspring_before_mutation von t-1 
            # 2. Mutation: species.genotype von t0 (oder crossover.offspring von t-1)
            # 3. Species: species von t0 
            # 4. Fitness: fitness_perspecies von t0
            # 5. Elimination: species von t0 mit crossover.parents von t0 abgleichen
            # 6. Crossover: crossover.parents von t0
            
            t0_cells = {}
            
            # Create a dict containing the uuid and species for all genotypes after mutation
            species_file_t0 = read_file(generation=gen, run=run, file="species")
            for i, species in enumerate(species_file_t0):
                for genotype in species.genotypes[1:]: # important! skip first genotype, because it doesnt exist in crossings_tm1
                    t0_cells[genotype.uuid] = {"species": i, "genotype_after_mutation": genotype}

            # Check if uuid will be parent or eliminated
            crossing_file_t0 = read_file(generation=gen, run=run, file="crossings")
            unique_parents_uuids_t0 = set([crossing[parent].uuid for crossing in crossing_file_t0.values() for parent in ["parent1", "parent2"]])
            for uuid in t0_cells.keys():
                t0_cells[uuid]["is_parent"] = uuid in unique_parents_uuids_t0 

            # ========================= Debugging (delete for final render)
            # cap t0_cells for faster rendering during development. Keep all parents and a few non_parents
            t0_parents = {uuid:genotype for uuid,genotype in t0_cells.items() if genotype["is_parent"]}
            t0_loosers = {uuid:t0_cells[uuid] for i, uuid in enumerate([uuid for uuid in t0_cells.keys() if not t0_cells[uuid]["is_parent"]]) if i < (debug_sample - len(t0_parents))}
            t0_cells = t0_parents | t0_loosers
            # ========================= Debugging

            # Get fitness of each genotype
            fitness_file_t0 = read_file(generation=gen, run=run, file="fitness_perspecies")
            fitness_list = {fitness_file_t0[species][1][i].uuid: fitness_file_t0[species][0][i] for species in fitness_file_t0 for i in range(len(fitness_file_t0[species][1]))}
            for uuid in t0_cells.keys():
                t0_cells[uuid]["fitness"] = fitness_list[uuid]
            fitness_range = (min(list(fitness_list.values())), max(list(fitness_list.values())))

            # Get the genotype before mutation
            if gen != 0:
                crossing_file_tm1 = read_file(generation=gen-1, run=run, file="crossings")    
                crossing_file_tm1_restructured = {crossing["offspring_before_mutation"].uuid: crossing for crossing in crossing_file_tm1.values()}            
                for uuid in t0_cells.keys():
                    t0_cells[uuid]["genotype_before_mutation"] = crossing_file_tm1_restructured[uuid]["offspring_before_mutation"]
                    t0_cells[uuid]["parents"] = (crossing_file_tm1_restructured[uuid]["parent1"].uuid, crossing_file_tm1_restructured[uuid]["parent2"].uuid)
                
                species_file_tm1 = read_file(generation=gen-1, run=run, file="species")
                species_per_genotype_uuid = {genotype.uuid: i for i, species in enumerate(species_file_tm1) for genotype in species.genotypes}
                for uuid in t0_cells.keys():
                    t0_cells[uuid]["parent_species"] = species_per_genotype_uuid[t0_cells[uuid]["parents"][0]] 


            # ==== Cell Creation ====
            crossover_animations = []
            for i, uuid in enumerate(t0_cells.keys()):                
                cell = species_shapes[t0_cells[uuid]["parent_species"] if gen != 0 else 0].copy().scale(random.uniform(0.9, 1.1))
                cell.move_to((
                    random.uniform(-8 + cell.get_width() + 0.075, 2.5 - cell.get_width() - 0.075), # + 0.075 for additional padding
                    random.uniform(-4.5 + cell.get_height() + 0.075, 3 - cell.get_height() - 0.075), 
                    0
                    ))
                t0_cells[uuid]["cell"] = cell

                if gen != 0:
                    parent1, parent2 = t0_cells[uuid]["parents"]
                    crossover_animations.append(Succession(
                        AnimationGroup(
                            tm1_cells[parent1]["cell"].animate.set_background_stroke(color=colors["crossover"], width=5),  
                            tm1_cells[parent2]["cell"].animate.set_background_stroke(color=colors["crossover"], width=5)
                            ),
                        AnimationGroup(GrowFromCenter(cell)),
                        AnimationGroup(
                            tm1_cells[parent1]["cell"].animate.set_background_stroke(color=self.camera.background_color, width=4),
                            tm1_cells[parent2]["cell"].animate.set_background_stroke(color=self.camera.background_color, width=4)
                            ),
                    ))
                else:
                    crossover_animations.append(GrowFromCenter(cell))


            # ==== Animations ==== 
            # == Crossover (v Initialisation if gen == 0) ==
            print(f"== Generation {gen:003} - Crossover")
            headline_texts = change_headline(gen, "Crossover" if gen != 0 else "Initialisation", headline_texts)
            self.add(*headline_texts)

            self.play(LaggedStart(
                *crossover_animations,
                lag_ratio=calc_lag_ratio(10, 3, len(t0_cells)),
                run_time=10
            ))

            # Parents perish
            if gen != 0:
                perish_animations = []
                for uuid in tm1_cells.keys():
                    perish_animations.extend([
                        tm1_cells[uuid]["cell"].animate.set_background_stroke(color=colors["elimination"]),
                        FadeOut(tm1_cells[uuid]["cell"]),
                    ])
                    
                self.play(
                    AnimationGroup(*perish_animations),
                    rate_func=rate_functions.ease_in_expo,
                    run_time=5
                )
        
            # == Mutation ==
            print(f"== Generation {gen:003} - Mutation")
            headline_texts = change_headline(gen, "Mutation", headline_texts)
            self.add(*headline_texts)

            mutation_animations = []
            for genotype in t0_cells.values():
                mutation_animations.append(Succession(
                    AnimationGroup(genotype["cell"].animate.set_background_stroke(color=colors["mutation"], width=5), run_time=1),
                    AnimationGroup(genotype["cell"].animate.set_background_stroke(color=self.camera.background_color, width=4), run_time=1),
                )) 
            # Animations cannot be sequentially composed when working on the same object, without wrapping each animation in an AnimationGroup
            # https://github.com/ManimCommunity/manim/issues/3338

            self.play(LaggedStart(
                *mutation_animations,
                lag_ratio=calc_lag_ratio(5, 2, len(t0_cells)),
                run_time=5,
            ))
            
            # == Speciation ==
            print(f"== Generation {gen:003} - Speciation")
            headline_texts = change_headline(gen, "Speciation", headline_texts)
            self.add(*headline_texts)

            if gen != 0:
                affected_uuids = [uuid for uuid in t0_cells.keys() if t0_cells[uuid]["parent_species"] != t0_cells[uuid]["species"]]
                new_cells = [species_shapes[t0_cells[uuid]["species"]].copy().scale(random.uniform(0.9, 1.1)).move_to(t0_cells[uuid]["cell"].get_center()) for uuid in affected_uuids]

                self.play(
                    AnimationGroup(*[Transform(t0_cells[uuid]["cell"], new_cell) for uuid, new_cell in zip(affected_uuids, new_cells)]),
                    rate_func=rate_functions.ease_in_expo,
                    run_time=5,
                    )
                    
                for uuid in affected_uuids:
                    t0_cells[uuid]["cell"] = new_cells[affected_uuids.index(uuid)]  # dict manipulation
            

            # == Fitness ==
            print(f"== Generation {gen:003} - Fitness Evaluation")
            headline_texts = change_headline(gen, "Fitness Evaluation", headline_texts)
            self.add(*headline_texts)

            fitness_animations = []
            for uuid in t0_cells.keys():
                relative_fitness_color = ManimColor(RED).interpolate(GREEN, (t0_cells[uuid]["fitness"] - fitness_range[0]) / (fitness_range[1] - fitness_range[0]))
                fitness_animations.append(t0_cells[uuid]["cell"].animate.set_fill(color=relative_fitness_color))
                
            self.play(
                *fitness_animations,
                run_time=5,
                )
            

            # == Elimination ==
            print(f"== Generation {gen:003} - Elimination")
            headline_texts = change_headline(gen, "Elimination", headline_texts)
            self.add(*headline_texts)

            elimination_animations = []
            for genotype in t0_cells.values():
                if not genotype["is_parent"]:
                    elimination_animations.append(Succession(
                        AnimationGroup(genotype["cell"].animate.set_background_stroke(color=colors["elimination"], width=5), run_time=1),
                        AnimationGroup(genotype["cell"].animate.set_opacity(0)),
                    ))

            self.play(LaggedStart(
                *elimination_animations,
                lag_ratio=calc_lag_ratio(5, 2, len(t0_cells)),
                run_time=5,
            ))

            # Put parent cells in tm1_cells dict, t0_cells will be reset in next iteration, so eliminated cells are discarded automatically
            tm1_cells = {uuid: info for uuid, info in t0_cells.items() if info["is_parent"]}  # dict manipulation


            self.wait(1)



class SceneForExperiments(Scene00Setup):
    def construct(self):
        super().construct()

        self.camera.frame.scale_to_fit_height(9)

        # Wobbling Rectangle
        num_points = 500
        wobble_frequency = 1
        num_waves = 7
        max_wobble_offset = 0.25
        width = 5.5 - max_wobble_offset*3
        height = 5.625 - max_wobble_offset*3

        rect = Rectangle(width=width, height=height).round_corners(radius=0.3).move_to((5.25, 1.6875, 0)).set_stroke(opacity=0.5)

        new_points = [rect.point_from_proportion(x/num_points) for x in range(num_points)]
        new_points[-1] = new_points[0]
        rect.clear_points()
        rect.set_points_smoothly(new_points)

        rect.reference = rect.copy().set_opacity(0)
        rect.time_ellapsed = 0

        sin_inputs = np.arange(0, num_waves*2*np.pi, (num_waves*2*np.pi) / rect.points.__len__())

        def wobble(mobject, dt):
            mobject.time_ellapsed += dt

            for i, point in enumerate(mobject.points):                    
                angle = 2*np.pi * i / mobject.points.__len__()
                amplitude = (max_wobble_offset/2) * np.sin(sin_inputs[i] + (wobble_frequency * mobject.time_ellapsed)) + (max_wobble_offset/2)  # oscelating between 0 and __max_wobble_offset
                point[0], point[1] = mobject.reference.points[i][0]+amplitude*np.cos(angle), mobject.reference.points[i][1]+amplitude*np.sin(angle)
            
            mobject.points[-1] = mobject.points[0]
            mobject.set_fill(color=WHITE, opacity=1)



        # Graph
        species_list = read_file(generation=0, run=run, file="species")
        genes = decypher_genotype(species_list[0].genotypes[0])
        nucleus = Nucleus(width, height, genes, use_case="simulation").move_to(rect.get_center())
        nucleus.set_style(fill_opacity=1, stroke_opacity=1, background_stroke_opacity=1)
        nucleus.adjust_edge_opacity()
        
        annotations = {
            "input":["x_coord", "y_coord", "x_velo", "y_velo", "angle", "ang_velo", "left_leg", "right_leg", "bias"], 
            "output": ["vert_thrust", "hor_thrust"]
        }
        for type, labels in annotations.items():
            annotations[type] = VGroup(*[CText(label).scale(0.2) for label in labels])
            annotations[type].arrange(DOWN, center=False, buff=0.382, aligned_edge=LEFT if type == "input" else RIGHT)

            x_coord = 5.25 - width/2 + max_wobble_offset*3 if type == "input" else 5.25 + width/2 - max_wobble_offset*3
            y_coord = 1.6875 if type == "input" else 1.44
            annotations[type].move_to((x_coord, y_coord, 0))


        with open("gymnasium_videos/gen75species2/log.json", "r") as f:
            node_activation = json.load(f)

        nucleus.frame = 1
        def node_updater(nucleus, dt):
            current_activation = node_activation[str(nucleus.frame)] 
            for idx, vertice in nucleus.vertices.items():
                if idx < 9:
                    activation = current_activation["input"][idx]
                else:
                    activation = current_activation["output"][idx-9]  # ! gym observations können > 1 sein (max in gen75species2 == 1.5)
                nucleus[idx].set_background_stroke(color=ManimColor(RED).interpolate(GREEN, activation))
            
            if nucleus.frame == len(node_activation):
                nucleus.remove_updater(node_updater)
            else:
                nucleus.frame += 1


        # Apply updater
        nucleus.add_updater(node_updater)
        # rect.add_updater(wobble)


        self.add(rect, nucleus, *annotations.values())
        self.wait(8)



        # objects = [
        #     Triangle(color=WHITE).round_corners(radius=0.3).scale(1.3).shift(RIGHT*0.5).make_smooth(),
        #     Square().round_corners(radius=1).scale(0.9).shift(LEFT*0.5),
        # ]
        # [obj.set_style(
        #     fill_color=YELLOW, 
        #     fill_opacity=1, 

        #     stroke_color="#222222",
        #     stroke_width=4,

        #     background_stroke_width=10,
        #     background_stroke_color=RED,

        #     ) for obj in objects]
        


        # self.add(
        #      # Square().round_corners(radius=0.5).scale(0.9).shift(DOWN*2),
        #     # RegularPolygon(n=6, color=WHITE).round_corners(radius=0.4).shift(UP*2+RIGHT),
        #     # RegularPolygon(n=5, color=WHITE, fill_color=BLUE).round_corners(radius=0.3).shift(UP*2+LEFT*2),
        # )



        # self.wait(1)
        # cell.nucleus.set_fill(opacity=0)
        # cell.nucleus.set_stroke(opacity=0)
        # self.wait(1)
        # cell.fitness.set_fill(opacity=1)

        # self.wait(1)


        # =========
        
        # species_list = read_file(generation=0, run=run, file="species")
        # genes = decypher_genotype(species_list[0].genotypes[0])
        # cell_height = 4
        # cell = Cell(width=cell_height/2, height=cell_height, genes=genes, num_waves=6, wobble_frequency=3)

        # angle = PI*1/4
        # direction = np.array([-np.cos(angle), np.sin(angle), 0])

        # self.play(
        #     Rotate(cell, angle),
        #     run_time=2
        # )

        # self.play(
        #     cell.animate.move_to(direction*2),
        #     run_time=2
        # )
