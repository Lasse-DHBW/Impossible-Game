
from manim import *
import numpy as np
from manim_utils import *
import json
import pandas as pd


class Particle(Ellipse):
    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        x_dir, y_dir = random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)
        self.velocity = np.array((x_dir*scale_factor, y_dir*scale_factor, 0))

    def move(self):
        self.shift(self.velocity)


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
    def __init__(self, cell_width, cell_height, genes, padding=None):
        node_genes, connection_genes = genes

        # Make node levels and innovation numbers contiguous whilst preserving order
        node_levels = sorted(set(node_genes["node_level"]))
        mapping_level = {old: new for old, new in zip(node_levels, range(node_levels.__len__()))}
        node_genes["node_level"] = node_genes["node_level"].map(mapping_level)

        innov_numbers = sorted(set(node_genes["innovation_number"]))
        mapping_innov = {old: new for old, new in zip(innov_numbers, range(innov_numbers.__len__()))}
        connection_genes["in_node"] = connection_genes["in_node"].map(mapping_innov)
        connection_genes["out_node"] = connection_genes["out_node"].map(mapping_innov)
        node_genes["innovation_number"] = node_genes["innovation_number"].map(mapping_innov)

        self.node_genes = node_genes.sort_values("innovation_number")
        self.connection_genes = connection_genes
        
        h_padding = cell_width*0.4 if padding is None else padding[0]*2
        v_padding = cell_height*0.4 if padding is None else padding[1]*2

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
                background_stroke_color="#EBA607", 
                background_stroke_width=8,
                background_stroke_opacity=0,
                stroke_opacity=1,
                stroke_width=2,
                stroke_color="#222222"
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
        enabled_genes =  self.connection_genes[self.connection_genes["is_disabled"] == False]
        weights = enabled_genes["weight"].to_numpy()
        
        # Normalize weights to the range [0, 1], maintaining the sign
        max_abs_weight = np.max(np.abs(weights))
        normalized_weights = weights / max_abs_weight
        
        # Opacity mapping: higher opacity for high absolute values, with a minimum opacity threshold
        def opacity_mapping(weight):
            return max(0.1, np.abs(weight))  # Ensures opacity is at least 0.1

        opacities = [opacity_mapping(weight) for weight in normalized_weights]

        # Set edge opacity based on calculated opacities
        for i, edge in enumerate(self.edges):
            self.edges[edge].set_opacity(opacities[i])

        
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

        new_row = pd.DataFrame({
            "innovation_number":[self.connection_genes["innovation_number"].max() + 1],
            "in_node": [in_node],
            "out_node": [out_node],
            "weight": [weight],
            "is_disabled": [is_disabled]
        })

        self.connection_genes = pd.concat([self.connection_genes, new_row], ignore_index=True)
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
        
        self.fitness_range = (-500, 0) 
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
        fitness_evaluation = (new_fitness - self.fitness_range[0]) / (self.fitness_range[1] - self.fitness_range[0])
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
        

            
class Closeup():
    def __init__(
            self, 
            run,
            gen, 
            num_points=400, 
            wobble_frequency=1, 
            num_waves=4, 
            max_wobble_offset=0.15, 
            disable_wobble=False
            ):
        self.gen = gen
        self.max_wobble_offset = max_wobble_offset

        self.width = 5.5 - max_wobble_offset*3
        self.height = 5.625 - max_wobble_offset*3

        self.membrane, self.wobble = self.create_membrane(num_points, wobble_frequency, num_waves, disable_wobble)
        self.labels = self.create_labels() 

        self.genotype_before_mutation, self.genotype_after_mutation = self.get_fittest_indv(run)
        self.nucleus_before_mutation = self.create_nucleus(self.genotype_before_mutation) if self.genotype_before_mutation is not None else None
        self.nucleus_after_mutation = self.create_nucleus(self.genotype_after_mutation)

    def get_fittest_indv(self, run):
        fitness_perspecies = read_file(generation=self.gen, run=run, file="fitness_perspecies")

        best_max_fitness = float('-inf')
        for species, (fitnesses, genotypes) in fitness_perspecies.items():
            max_fitness = max(fitnesses)
            max_index = fitnesses.index(max_fitness)

            if max_fitness > best_max_fitness:
                best_max_fitness = max_fitness
                self.best_species = species
                genotype_after_mutation = genotypes[max_index]
                self.fittest_indv_uuid = genotype_after_mutation.uuid

        genotype_before_mutation = None
        if self.gen != 0:
            crossovers = read_file(generation=self.gen-1, run=run, file="20240104_225458_crossings_new")
            crossovers_restructures = {crossover["offspring"].uuid:crossover for crossover in crossovers.values()}
            genotype_before_mutation = crossovers_restructures[genotype_after_mutation.uuid]["offspring_before_mutation"]                    

        return genotype_before_mutation, genotype_after_mutation
    
    def create_membrane(self, num_points, wobble_frequency, num_waves, disable_wobble):
        rect = Rectangle(width=self.width, height=self.height).round_corners(radius=0.75).move_to((5.25, 1.6875, 0)).set_stroke(opacity=0.5, width=3)
        new_points = [rect.point_from_proportion(x/num_points) for x in range(num_points)]
        new_points[-1] = new_points[0]
        rect.clear_points()
        rect.set_points_smoothly(new_points)

        sin_inputs = np.arange(0, num_waves*2*np.pi, (num_waves*2*np.pi) / rect.points.__len__())
        rect.reference = rect.copy().set_opacity(0)
        rect.time_ellapsed = 0

        def wobble(mobject, dt):
            mobject.time_ellapsed += dt

            for i, point in enumerate(mobject.points):                    
                angle = 2*np.pi * i / mobject.points.__len__()
                amplitude = (self.max_wobble_offset/2) * np.sin(sin_inputs[i] + (wobble_frequency * mobject.time_ellapsed)) + (self.max_wobble_offset/2)  # oscelating between 0 and __max_wobble_offset
                point[0], point[1] = mobject.reference.points[i][0]+amplitude*np.cos(angle), mobject.reference.points[i][1]+amplitude*np.sin(angle)
            
            mobject.points[-1] = mobject.points[0]
            # mobject.set_fill(color=WHITE, opacity=1)

        if not disable_wobble:
            rect.add_updater(wobble)

        return rect, wobble

    def create_labels(self):
        texts = ["bias =", "x_coord =", "y_coord =", "x_velo =", "y_velo =", "angle =", "ang_velo =", "l_leg =", "r_leg =", "= v_thrst", "= h_thrst"]
        labels = []
        for text in texts:
            labels.append((
                CText(text).scale(0.2),
                DecimalNumber(0.00, num_decimal_places=2, include_sign=True).scale(0.29)
                ))

        # Arrange pairs of texts and numbers horizontally
        input_labels = [VGroup(*label).arrange(RIGHT, buff=0.05) for label in labels[:9]]
        output_labels = [VGroup(*label).arrange(LEFT, buff=0.05) for label in labels[9:]]
        
        # Arrange all input and output pairs vertically
        input_labels = VGroup(*input_labels).arrange(UP, center=False, buff=0.4, aligned_edge=LEFT)  # arrange up to account for node numbering
        output_labels = VGroup(*output_labels).arrange(UP, center=False, buff=0.4, aligned_edge=RIGHT)

        # Position input and output
        input_labels.move_to((5.25 - self.width/2 + input_labels.get_width()/2 + self.max_wobble_offset*1.5, 1.6875, 0))
        output_labels.move_to((5.25 + self.width/2 - output_labels.get_width()/2 - self.max_wobble_offset*1.5, 1.44, 0))

        # Group input and output
        labels = VGroup(input_labels, output_labels)

        return labels

    def create_nucleus(self, genotype):
        genes = decypher_genotype(genotype)
        h_padding = self.max_wobble_offset*1.5 + max([vrow.get_width() for vrow in self.labels]) + 0.1 + 0.025 # 0.1 = node radius,
        v_padding = (self.height - max([vrow.get_height() for vrow in self.labels]))/2 + 0.1 - 0.05 # 0.5  manual fine tuning
        nucleus = Nucleus(self.width, self.height, genes, padding=(h_padding, v_padding)).move_to(self.membrane.get_center())
        nucleus.set_style(fill_opacity=1, stroke_opacity=1, background_stroke_opacity=1)
        nucleus.adjust_edge_opacity()

        return nucleus
    
    def launch_lunar_lander(self, best_species):
        with open(f"gymnasium_videos/20240104_225458/gen{self.gen}species{best_species}/log.json", "r") as f:
            node_activation = json.load(f)

        self.frame = 1
        vertice_nums = self.nucleus_after_mutation.vertices.keys()
        in_out_vertice_nums = list(vertice_nums)[:9] + list(vertice_nums)[-2:]
        def node_updater(nucleus, dt):
            current_activation = node_activation[str(self.frame)] 
            for idx in in_out_vertice_nums:
                if idx < 9:
                    self.labels[0][idx][1].set_value(current_activation["input"][idx])
                    self.labels[0][idx][1].move_to(self.labels[0][idx][1].get_center()+UP*0.0001)
                    activation = current_activation["input"][idx]
                    nucleus[idx].set_background_stroke(color=ManimColor(RED).interpolate(GREEN, activation))
                elif idx >= 9:
                    i = 0 if idx == in_out_vertice_nums[-2] else 1
                    self.labels[1][i][1].set_value(current_activation["output"][i])
                    activation = current_activation["output"][i]  # ! gym observations können > 1 sein (max in gen75species2 == 1.5)
                    nucleus[idx].set_background_stroke(color=ManimColor(RED).interpolate(GREEN, activation))

            if self.frame == len(node_activation):
                nucleus.remove_updater(node_updater)
            else:
                self.frame += 1

        self.nucleus_after_mutation.add_updater(node_updater)
        return node_activation.__len__()  # return number of frames to calc required waiting time



class ColorChangeAnimation(Animation):
    # Needs to be custom build so that the membrane can change color while wobbling
        
    def __init__(self, mobject, target_color, wobble_updater, **kwargs):
        super().__init__(mobject, **kwargs)
        self.target_color = ManimColor(target_color)
        self.wobble_updater = wobble_updater

    def interpolate_mobject(self, alpha):
        current_color = self.mobject.get_color()
        new_color = interpolate_color(current_color, self.target_color, alpha)
        self.mobject.set_color(new_color)

        # Apply wobble updater with the calculated dt
        self.wobble_updater(self.mobject, 1/50)
