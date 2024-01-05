
from manim import *
import numpy as np
from manim_utils import CText

class Membrane(VMobject):
    def __init__(self, reference_ellipse, num_waves, max_wobble_offset, wobble_frequency):
        super().__init__()

        # For membrane shape
        self.reference_ellipse = reference_ellipse
        self.set_points(self.reference_ellipse.points.copy())

        # For style
        self.set_style(
            stroke_color=WHITE,
            stroke_opacity=1,
            fill_opacity=0
        )

        # Wobble Animation
        self.__counter = 0
        self.__time_ellapsed = 0

        self.__max_wobble_offset = max_wobble_offset
        self.wobble_frequency = wobble_frequency
        self.num_waves = num_waves  # how many waves circle the membrane
        self.__sin_inputs = np.arange(0, self.num_waves*2*np.pi, (self.num_waves*2*np.pi) / self.points.__len__())

        self.add_updater(self.wobble)

    def wobble(self, mobject, dt):
        self.__time_ellapsed += dt

        for i, point in enumerate(self.points):                    
            angle = 2*np.pi * i / self.points.__len__()
            amplitude = (self.__max_wobble_offset/2) * np.sin(self.__sin_inputs[i] + (self.wobble_frequency * self.__time_ellapsed)) + (self.__max_wobble_offset/2)  # oscelating between 0 and __max_wobble_offset
            point[0], point[1] = self.reference_ellipse.points[i][0]+amplitude*np.cos(angle), self.reference_ellipse.points[i][1]+amplitude*np.sin(angle)
        
        self.points[-1] = self.points[0]
        self.__counter = self.__counter + 1

  
class Nucleus(Graph):
    def __init__(self, cell_width, cell_height, genes, use_case="explainer"):
        self.node_genes, self.connection_genes = genes
        
        v_padding = cell_height*0.4 if use_case == "explainer" else 1
        h_padding = cell_width*0.4 if use_case == "explainer" else 0.4

        self.layout = self.create_layout(
            height=(cell_height-v_padding), 
            spacing=(cell_width-h_padding)/self.partitions.__len__(), 
            partitions=self.partitions
            )

        vertex_mobjects = {}
        for node in self.node_set:
            vertex_mobjects[node] = LabeledDot(
                CText(f"{node}", color=DARK_GRAY, font_size=10),
                radius=0.10, 
                background_stroke_color=YELLOW_D, 
                background_stroke_width=8,
                background_stroke_opacity=0,
                stroke_opacity=1,
                stroke_width=2,
                stroke_color=BLACK
                )

        super().__init__(
            self.node_set,
            self.edge_set,
            layout=self.layout,
            labels=True,
            vertex_mobjects=vertex_mobjects,
            edge_config={
                "stroke_width": 2,
                "color": WHITE,
            },
        )

    @property
    def node_set(self):
        return self.node_genes["innovation_number"].to_numpy()

    @property
    def edge_set(self):
        return [(row[1]['in_node'], row[1]['out_node']) for row in self.connection_genes.iterrows() if row[1]['is_disabled'] == False]

    @property
    def partitions(self):
        return [list(self.node_genes[self.node_genes['node_level'] == i]['innovation_number']) for i in range(self.node_genes['node_level'].max() + 1)]

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
        max_nodes_in_layer = max([len(layer) for layer in partitions])

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
        
    def get_visual_genotype(self, font_scaling):
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

                texts.append(CText(f"Node {node}").scale(font_scaling))
                texts.append(CText(f"{layer}").scale(font_scaling))

                save_to_dict("node_genes", texts)

            
        # Create connection genes
        for edge in self.connection_genes.iterrows():
            texts = []

            texts.append(CText(f"In {edge[1]['in_node']}").scale(font_scaling))
            texts.append(CText(f"Out {edge[1]['out_node']}").scale(font_scaling))
            texts.append(VGroup(
                CText(f"Weight ").scale(font_scaling),
                DecimalNumber(edge[1]['weight'], num_decimal_places=2).scale(font_scaling*(4/3))
            ).arrange(RIGHT, buff=0.05))
            # texts.append(CText(f"Weight {edge[1]['weight']:.2f}").scale(font_scaling))
            texts.append(CText(f"{'Disabled' if edge[1]['is_disabled'] else 'Enabled'}").scale(font_scaling))
            texts.append(CText(f"Innov {edge[1]['innovation_number']}").scale(font_scaling))

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


class Cell(VGroup):
    def __init__(self, width, height, genes, fitness, num_waves, wobble_frequency):

        max_wobble_offset = 0.1 # modulates max amplitude of wobble animation

        # The reference ellipse is a hidden ellipse that is always moving whenever the cell is moving
        # It acts as a reference for the membrane to wobble around
        ellipse_points= self._create_ellipse_points(height - max_wobble_offset, width - max_wobble_offset)
        self.__reference_ellipse = VMobject()
        self.__reference_ellipse.set_points_smoothly(ellipse_points)
        self.__reference_ellipse.set_opacity(0)
        
        self.membrane = Membrane(
            reference_ellipse=self.__reference_ellipse,
            num_waves=num_waves,
            max_wobble_offset=max_wobble_offset,
            wobble_frequency=wobble_frequency,
            )
        
        # Create Nucleus - This is the neural network inside the cell
        self.nucleus = Nucleus(
            cell_width = width,
            cell_height = height,
            genes=genes,
            )
        
        self.fitness_range = (-500, -283) 
        self.fitness = DecimalNumber(0, num_decimal_places=0, font_size=60, fill_opacity=0)
        self.fitness.rotate(PI/2, about_point=self.fitness.get_center())
        self.fitness.move_to((0, 0, 0))
        self.update_fitness(fitness)

        super().__init__(
            self.__reference_ellipse,
            self.membrane,
            self.nucleus,
            self.fitness
        )


    def update_fitness(self, new_fitness):
        self.fitness.set_value(new_fitness).rotate(PI/2, about_point=self.fitness.get_center())  # Rotate need to be reapplied
        fitness_evaluation = new_fitness / (self.fitness_range[1] - self.fitness_range[0])
        self.fitness.set_fill(color=ManimColor(RED).interpolate(GREEN, fitness_evaluation))

    def _create_ellipse_points(self, height, width):
        points = []
        for i in range(200):
            angle = 2 * np.pi * i / 200
            x = width / 2 * np.cos(angle)
            y = height / 2 * np.sin(angle)
            points.append([x, y, 0])

        points.append(points[0])  # close the circle
        return points
        