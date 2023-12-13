
from manim import *
import numpy as np

class Membrane(VMobject):
    def __init__(self, cell, color, width=4, height=2, num_points=200):
        super().__init__()
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
        self.time_ellapsed = 0

        # For rotation
        self.rotation_target = 0
        self.rotation_current = 0
        self.rotation_speed = 0.01


    def toggle_wobble(self, enable=True):
        if enable:
            self.add_updater(self.wobble)
            self.add_updater(self.follow_cell)
        else:
            self.clear_updaters()

    def __create_points(self):
        points = []
        for i in range(self.__num_points + 1):
            angle = 2 * np.pi * i / self.__num_points
            x = self.__w / 2 * np.cos(angle)
            y = self.__h / 2 * np.sin(angle)
            points.append([x, y, 0])

        self.set_points_smoothly(points) 

    def wobble(self, mobject, dt):
        self.time_ellapsed += dt
        for i, point in enumerate(mobject.points):                    
            amplitude = self.__wobble_variance * np.sin(self.__wobble_wavelength*(self.__counter + i)) + self.__wobble_mw
            factor = amplitude * np.sin(self.__wobble_frequency * self.time_ellapsed) + 1
            point[0], point[1] = self.__original_coords[i][0]*factor, self.__original_coords[i][1]*factor
        
        mobject.points[-1] = mobject.points[0]
        self.__counter = self.__counter + 1

    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())
        if self.rotation_target > self.rotation_current:
            self.rotation_current += self.rotation_speed
            self.rotate(self.rotation_current)

  
class Nucleus(Graph):
    def __init__(self, cell, color, genes):
        vertices, edges, partitions = genes

        super().__init__(
            vertices,
            edges,
            layout='partite',
            partitions=partitions,
            layout_scale=2,
            # vertex_config={'radius': 0.20},
        )

        self.rotate(PI/2)

        # For movement
        self.cell = cell
        self.rotation_target = 0
        self.rotation_speed = 0.01
        self.add_updater(self.follow_cell)


    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())
        if self.rotation_target > 0:
            self.rotate(self.rotation_speed)
            self.rotation_target -= self.rotation_speed
        

class Cell(Rectangle):
    def __init__(self, width, height, genes, color, **kwargs):
        super().__init__(width=width, height=height, color=color, **kwargs)

        self.nucleus = Nucleus(cell=self, color=color, genes=genes)
        self.membrane = Membrane(cell=self, color=color)
    
    def rotate_all(self, angle):
        self.nucleus.rotation_target = angle
        self.membrane.rotation_target = angle
        self.rotate(angle)
