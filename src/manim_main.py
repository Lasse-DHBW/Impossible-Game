from manim import *
from manim_utils import get_generation_info, decypher_genotype
from manim_mobjects import Cell
import random


# Debugging
render_animations = True
random.seed = 42

# Aesthetics
font_family = "Montserrat"
font_color = "#eeeeee"

# Neat
run = '2023_12_07_19_59_20_340927'

class Scene00Setup(MovingCameraScene):
    def play(self, *args, **kwargs):
        if render_animations:
            super().play(*args, **kwargs)

    def construct(self):
        self.camera.background_color = "#222222"


class Scene01ColdOpener(Scene00Setup):
    def construct(self):
        super().construct()

        # TODO: Implement


class Scene02Title(Scene00Setup):
    def construct(self):
        super().construct()

        title = Text("NEAT", font=font_family, weight=BOLD, color=font_color).shift(UP).scale(2)
        subtitle = Text("NeuroEvolution of Augmenting Topologies", font=font_family, color=font_color).scale(0.8)

        self.play(Write(title))
        self.wait(1)
        self.play(Write(subtitle))


class Scene03Individual(Scene00Setup):
    def construct(self):
        super().construct()

        species_list, fitness_per_species = get_generation_info(generation=0, run=run)
        genes = decypher_genotype(species_list[0].representative)
        cell = Cell(width=2, height=2, genes=genes, color=WHITE)

        self.play(GrowFromCenter(cell.membrane, run_time=1), FadeIn(cell.nucleus, run_time=2))
        self.wait(1)


class TestScene(MovingCameraScene):
    def play(self, *args, **kwargs):
        if render_animations:
            super().play(*args, **kwargs)

    def construct(self):
        pass
