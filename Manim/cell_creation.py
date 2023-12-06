from manim import *
import numpy as np

class Membrane(VMobject):
    def __init__(self, scene, cell, width=4, height=2, num_points=200, **kwargs):
        super().__init__(**kwargs)
        self.scene = scene
        self.cell = cell

        # For membrane shape
        self.__w = width
        self.__h = height
        self.__num_points = num_points
        self.__create_points()

        # Wobble Animation
        self.__original_coords = self.points.copy()
        self.__counter = 0
        self.__wobble_frequency = 5
        self.__wobble_mw = 0.05
        self.__wobble_variance = 0.025
        self.__wobble_wavelength = 0.04

        self.add_updater(self.wobble)

    def __create_points(self):
        points = []
        for i in range(self.__num_points + 1):
            angle = 2 * np.pi * i / self.__num_points
            x = self.__w / 2 * np.cos(angle)
            y = self.__h / 2 * np.sin(angle)
            points.append([x, y, 0])

        self.set_points_smoothly(points) 

    def wobble(self, mobject, dt):
        for i, point in enumerate(mobject.points):                    
            amplitude = self.__wobble_variance * np.sin(self.__wobble_wavelength*(self.__counter + i)) + self.__wobble_mw
            factor = amplitude * np.sin(self.__wobble_frequency * self.scene.renderer.time) + 1
            point[0], point[1] = self.__original_coords[i][0]*factor, self.__original_coords[i][1]*factor
        
        mobject.points[-1] = mobject.points[0]
        self.move_to(self.cell.get_center())
        self.__counter = self.__counter + 1

  
class Nucleus(Square):
    def __init__(self, cell, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.add_updater(self.follow_cell)
    
    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())


class MyScene(MovingCameraScene):
    def construct(self):
    
        cell = Rectangle(width=4.25, height=2.25)
        nucleus = Nucleus(cell=cell, side_length=0.5)
        membrane = Membrane(scene=self, color=WHITE, cell=cell)

        self.add(cell, nucleus, membrane)
        self.play(cell.animate.shift(RIGHT*2), run_time=3)

        