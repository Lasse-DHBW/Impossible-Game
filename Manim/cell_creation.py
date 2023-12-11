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

        # For rotation
        self.rotation_target = 0
        self.rotation_current = 0
        self.rotation_speed = 0.01

        self.add_updater(self.wobble)
        self.add_updater(self.follow_cell)

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
        self.__counter = self.__counter + 1

    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())
        if self.rotation_target > self.rotation_current:
            self.rotation_current += self.rotation_speed
            self.rotate(self.rotation_current)

  
class Nucleus(VGroup):
    def __init__(self, cell, genotype, **kwargs):   
        super().__init__(**kwargs)

        # For movement
        self.cell = cell
        self.rotation_target = 0
        self.rotation_speed = 0.01
        self.add_updater(self.follow_cell)

        # For Content
        self.genotype = genotype
        self.create_phenotype()

    def follow_cell(self, mobject, dt):
        self.move_to(self.cell.get_center())
        if self.rotation_target > 0:
            self.rotate(self.rotation_speed)
            self.rotation_target -= self.rotation_speed

    def create_phenotype(self):
        nodes = []
        edges = []
        # for gene in self.genotype:
        #     # TODO
        #     pass

        input_layer = Circle(radius=0.1, color=WHITE)
        output_layer = Circle(radius=0.1, color=WHITE).shift(RIGHT)
        arrow = Arrow(input_layer.get_center(), output_layer.get_center())
        self.add(input_layer, output_layer, arrow)
        self.center()
        

            

class Cell(Rectangle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def rotate_all(self, angle, nucleus, membrane):
        nucleus.rotation_target = angle
        membrane.rotation_target = angle
        self.rotate(angle)


class MyScene(MovingCameraScene):
    def construct(self):
        # Settings
        frame_height = 16
        frame_width = 28

        # Camera
        # self.camera.frame_height=frame_height   # default: 8
        # self.camera.frame_width=frame_width    # default:14
        
        # Background
        # numberplane = NumberPlane(
        #     x_range=(- frame_width//2, frame_width//2, 1), 
        #     y_range=(- frame_height//2, frame_height//2, 1)
        #     )
        # self.add(numberplane)
    
        # Cells
        cell = Cell(width=4.25, height=2.25)
        nucleus = Nucleus(cell=cell, genotype=None)
        membrane = Membrane(cell=cell, scene=self, color=WHITE)
        # cell.rotate_all(angle=np.pi, nucleus=nucleus, membrane=membrane)

        self.add(cell, nucleus, membrane)
        # self.play(cell.animate.shift(RIGHT*2, UP), run_time=3)
        self.wait(5)

        