# Aesthetics
background_color = "#333333"
primary_color = None
secondary_color = None



from manim import *


class testing(Scene):
    def construct(self):


        square = Square(fill_opacity=1).shift(UP)
        triangle = Triangle(color="#123456", fill_opacity=1).shift(RIGHT * 2)
        logo = VGroup(triangle, square)  # order matters


        self.camera.background_color = "#ece6e2"
        self.add(logo)  # puts element on canvas without animation


        dot1 = Dot([-2, -1, 0])  # Dots
        dot2 = dot1.copy().shift(4*RIGHT, 2*UP)  # copy elements
        line = Line(dot1.get_center(), dot2.get_center()).set_color(ORANGE)  # lines
        self.add(line, dot1)
        self.play(Transform(dot1, dot2))  # transition from one element to the next


        arrow = Arrow(ORIGIN, dot1, buff=0)  # arrows
        numberplane = NumberPlane() # matrix background
        self.add(arrow, numberplane)


        circle = Circle(radius=1, color=BLUE)
        self.play(GrowFromCenter(circle)) # puts element on canvas with animation

        mobject.match_coordinates()


        render_animations = True
        frame_height = 36 # default: 8
        frame_width = 63 # default:14

        def play(self, *args, **kwargs):
            if render_animations:
                super().play(*args, **kwargs)


            self.camera.frame_height=frame_height   # default: 8
            self.camera.frame_width=frame_width    # default:14
            
            numberplane = NumberPlane(
                x_range=(- frame_width//2, frame_width//2, 1), 
                y_range=(- frame_height//2, frame_height//2, 1)
                )
            
    membrane.set_style(
        stroke_color=WHITE,
        stroke_width=4,
        fill_color=BLUE,
        fill_opacity=0.5
    )