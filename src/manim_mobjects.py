
from manim import *
import numpy as np
from manim_utils import CText

class Membrane(VMobject):
    def __init__(self, cell, color, num_waves, wobble_frequency):
        super().__init__()
        self.cell = cell

        # For membrane shape
        self.__w = cell.width - cell.max_wobble_offset 
        self.__h = cell.height - cell.max_wobble_offset 
        self.__create_points()  # num points increases, because of set_points_smoothly
        self.__original_coords = self.points.copy()
    
        # Wobble Animation
        self.__counter = 0
        self.time_ellapsed = 0

        self.__max_wobble_offset = cell.max_wobble_offset
        self.wobble_frequency = wobble_frequency
        self.__num_waves = num_waves  # how many waves circle the membrane
        self.__sin_inputs = np.arange(0, self.__num_waves*2*np.pi, (self.__num_waves*2*np.pi) / self.points.__len__())
 
        # For rotation
        self.rotation_target = 0
        self.rotation_current = 0
        self.rotation_speed = 0.01

        self.add_updater(self.wobble)
        self.add_updater(self.follow_cell)

    def __create_points(self):
        points = []
        for i in range(200):
            angle = 2 * np.pi * i / 200
            x = self.__w / 2 * np.cos(angle)
            y = self.__h / 2 * np.sin(angle)
            points.append([x, y, 0])

        points.append(points[0])  # close the circle
        self.set_points_smoothly(points) # adds additional points to the 201 points

    def wobble(self, mobject, dt):
        self.time_ellapsed += dt

        for i, point in enumerate(self.points):                    
            angle = 2*np.pi * i / self.points.__len__()
            amplitude = (self.__max_wobble_offset/2) * np.sin(self.__sin_inputs[i] + (self.wobble_frequency * self.time_ellapsed)) + (self.__max_wobble_offset/2)  # oscelating between 0 and __max_wobble_offset
            point[0], point[1] = self.__original_coords[i][0]+amplitude*np.cos(angle), self.__original_coords[i][1]+amplitude*np.sin(angle)
        
        self.points[-1] = self.points[0]
        self.__counter = self.__counter + 1

    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())
        if self.rotation_target > self.rotation_current:
            self.rotation_current += self.rotation_speed
            self.rotate(self.rotation_current)

  
class Nucleus(Graph):
    def __init__(self, cell, color, genes):
        self.node_genes, self.connection_genes = genes
        
        self.layout = self.create_layout(
            height=(cell.height-cell.height*0.4), 
            spacing=(cell.width-cell.width*0.4)/self.partitions.__len__(), 
            partitions=self.partitions
            )

        labels = {}
        for node in self.node_set:
            labels[node] = CText(f"{node}", color=ManimColor("#000000")).scale(0.2)

        super().__init__(
            self.node_set,
            self.edge_set,
            layout=self.layout,
            labels=labels,
            vertex_config={'radius': 0.10},
        )

        for labeled_dot in self.vertices.values():
            labeled_dot.set_color(BLACK)
            labeled_dot.set_fill(color=WHITE)



        # self.set_fill(opacity=0)
        self.set_stroke(width=2)

        self.adjust_edge_opacity()

        # Rotate if necessary
        if cell.width > cell.height:
            self.rotate(PI/2)

        # For movement
        self.cell = cell
        self.rotation_target = 0
        self.rotation_speed = 0.01
        self.add_updater(self.follow_cell)

    @property
    def node_set(self):
        return self.node_genes["innovation_number"].to_numpy()

    @property
    def edge_set(self):
        return [(row[1]['in_node'], row[1]['out_node']) for row in self.connection_genes.iterrows()]

    @property
    def partitions(self):
        return [list(self.node_genes[self.node_genes['node_level'] == i]['innovation_number']) for i in range(self.node_genes['node_level'].max() + 1)]

    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())
        if self.rotation_target > 0:
            self.rotate(self.rotation_speed)
            self.rotation_target -= self.rotation_speed

    @staticmethod
    def create_layout(height, spacing, partitions):
        """
        Generate a layout for a neural network graph.

        :param height: The total height for the graph layout.
        :param spacing: The horizontal distance between each layer.
        :param partitions: A list of lists, where each sublist represents a layer in the neural network.
        :return: A dictionary representing the node positions in the layout.
        """
        layout = {}
        num_layers = len(partitions)
        max_nodes_in_layer = max(len(layer) for layer in partitions)

        # Calculate vertical positions
        if max_nodes_in_layer > 1:
            vertical_spacing = height / (max_nodes_in_layer - 1)
            vertical_positions = [-height / 2 + i * vertical_spacing for i in range(max_nodes_in_layer)]
        else:
            vertical_positions = [0]

        for i, layer in enumerate(partitions):
            layer_width = i * spacing - (spacing * (num_layers - 1) / 2)  # Position each layer evenly based on spacing

            # Center nodes in each layer vertically
            start_index = (max_nodes_in_layer - len(layer)) // 2
            for j, node in enumerate(layer):
                node_height = vertical_positions[start_index + j]

                layout[node] = [layer_width, node_height, 0]

        return layout
        
    def adjust_edge_opacity(self):
        max_weight = self.connection_genes["weight"].max()
        weights = self.connection_genes["weight"].to_numpy()

        # Set edge opacity based on weight
        for i, edge in enumerate(self.edges):
            self.edges[edge].set_opacity(weights[i]/max_weight)
        
    def get_visual_genotype(self, scale_factor):
        """Returns the node genes and connection genes of the graph as boxes containing texts like in the paper."""
        visual_genotype = {
            "node_genes": [],
            "connection_genes": [],
        }
        def save_to_dict(gene_type, texts):
            # for i, text in enumerate(texts):
            #     if i != 0:
            #         text.next_to(texts[i-1], DOWN, buff=0.1)

            texts = VGroup(*texts).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            box = Rectangle(
                width=texts.width + 0.15, 
                height=texts.height + 0.15,
                stroke_width=1,
                ).move_to(texts.get_center())
            
            visual_genotype[gene_type].append(VGroup(texts, box))

        # Create node genes
        for idx, layer in enumerate(self.partitions):
            for node in layer:
                texts = []

                if idx == 0:
                    layer = "Input Layer"
                elif idx == self.partitions.__len__() - 1:
                    layer = "Output Layer"
                else:
                    layer = f"Hidden Layer {idx}"

                texts.append(CText(f"Node {node}").scale(scale_factor))
                texts.append(CText(f"{layer}").scale(scale_factor))

                save_to_dict("node_genes", texts)

            
        # Create connection genes
        for edge in self.connection_genes.iterrows():
            texts = []

            texts.append(CText(f"In {edge[1]['in_node']}").scale(scale_factor))
            texts.append(CText(f"Out {edge[1]['out_node']}").scale(scale_factor))
            texts.append(CText(f"Weight {edge[1]['weight']:.2f}").scale(scale_factor))
            texts.append(CText(f"{'Disabled' if edge[1]['is_disabled'] else 'Enabled'}").scale(scale_factor))
            texts.append(CText(f"Innov {edge[1]['innovation_number']}").scale(scale_factor))

            save_to_dict("connection_genes", texts)

        return visual_genotype

    def add_edge(self, in_node, out_node, weight, is_disabled):
        super().add_edges(((in_node, out_node)))

        self.connection_genes = self.connection_genes.append({
            "innovation_number":self.connection_genes["innovation_number"].max() + 1,
            "in_node": in_node,
            "out_node": out_node,
            "weight": weight,
            "is_disabled": is_disabled
        }, ignore_index=True)

        self.adjust_edge_opacity()


class Cell(Rectangle):
    def __init__(self, width, height, genes, color, num_waves, wobble_frequency, opacity):
        super().__init__(width=width, height=height, color=color)
        self.set_stroke(opacity=opacity)
        
        self.width = width
        self.height = height
        self.max_wobble_offset = 0.1

        self.nucleus = Nucleus(
            cell=self, 
            color=color,
            genes=genes,
            ).set_opacity(opacity)
        
        self.membrane = Membrane(
            cell=self, 
            color=color, 
            num_waves=num_waves,
            wobble_frequency=wobble_frequency,
            ).set_stroke(opacity=opacity)

    def rotate_all(self, angle):
        self.nucleus.rotation_target = angle
        self.membrane.rotation_target = angle
        self.rotate(angle)
