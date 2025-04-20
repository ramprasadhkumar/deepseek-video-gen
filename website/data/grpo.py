from manim import *
import numpy as np

class PythagoreanTheoremScene(Scene):
    def construct(self):
        # 1. Show title
        title = Text("Pythagorean Theorem").to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 2. Define triangle vertices for sides a=5, b=12, c=13
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([5.0, 0.0, 0.0])
        C = np.array([5.0, 12.0, 0.0])

        # center and scale triangle
        centroid = (A + B + C) / 3
        A -= centroid; B -= centroid; C -= centroid
        scale_factor = 0.4
        A *= scale_factor; B *= scale_factor; C *= scale_factor

        # 3. Draw the right triangle
        triangle = Polygon(A, B, C, color=WHITE)
        labels = VGroup(
            MathTex("a=5").scale(0.7).next_to(Line(A, B), DOWN, buff=0.1),
            MathTex("b=12").scale(0.7).next_to(Line(B, C), RIGHT, buff=0.1),
            MathTex("c=13").scale(0.7).next_to(Line(A, C), UP, buff=0.1),
        )
        self.play(Create(triangle), Write(labels))
        self.wait(1)

        # 4. Construct square on side a (blue)
        n_ab = np.array([0.0, 1.0, 0.0])
        len_ab = np.linalg.norm(B - A)
        square_a = Polygon(
            A,
            B,
            B + n_ab * len_ab,
            A + n_ab * len_ab,
            color=BLUE, fill_opacity=0.3
        )
        text_a2 = MathTex("a^2").scale(0.7).set_color(BLUE).next_to(square_a, DOWN, buff=0.1)
        self.play(Create(square_a), Write(text_a2))
        self.wait(1)

        # 5. Construct square on side b (green)
        n_bc = np.array([-1.0, 0.0, 0.0])
        len_bc = np.linalg.norm(C - B)
        square_b = Polygon(
            B,
            C,
            C + n_bc * len_bc,
            B + n_bc * len_bc,
            color=GREEN, fill_opacity=0.3
        )
        text_b2 = MathTex("b^2").scale(0.7).set_color(GREEN).next_to(square_b, RIGHT, buff=0.1)
        self.play(Create(square_b), Write(text_b2))
        self.wait(1)

        # 6. Construct square on hypotenuse c (red)
        v_ac = C - A
        n_ac = np.array([-v_ac[1], v_ac[0], 0.0]) / np.linalg.norm(v_ac)
        len_ac = np.linalg.norm(C - A)
        square_c = Polygon(
            A,
            C,
            C + n_ac * len_ac,
            A + n_ac * len_ac,
            color=RED, fill_opacity=0.3
        )
        text_c2 = MathTex("c^2").scale(0.7).set_color(RED).next_to(square_c, UP, buff=0.1)
        self.play(Create(square_c), Write(text_c2))
        self.wait(1)

        # 7. Show theorem statement
        theorem = MathTex("a^2 + b^2 = c^2").scale(0.8).to_edge(DOWN)
        self.play(Write(theorem))
        self.wait(2)