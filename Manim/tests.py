from manim import *
import numpy as np

class MyScene(MovingCameraScene):
    def construct(self):
    
        square = Square()
        self.add(square)

        for i in range(5):
            square.rotate((np.pi*2)/5)
            self.wait(1)