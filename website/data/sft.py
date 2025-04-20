# Import Manim's modules
from manim import *
import numpy as np # Import numpy for coordinate arrays

class PythTheorem(Scene):
    def construct(self):
        # --- Create a right triangle with legs 5 and 12 ---
        # The right angle is placed at the origin (0,0,0).
        # Vertices: (0,0), (5,0), (0,12)
        v_origin = np.array([0, 0, 0])
        v_side_a = np.array([5, 0, 0])
        v_side_b = np.array([0, 12, 0])

        triangle = Polygon(v_origin, v_side_a, v_side_b, color=WHITE, fill_opacity=0.1)
        triangle.set_stroke(width=3)

        # --- Create squares on each side of the triangle ---

        # Square on side 'a' (length 5) - along the x-axis
        # Vertices: (0,0), (5,0), (5,-5), (0,-5)
        square_a_verts = [v_origin, v_side_a, v_side_a + DOWN * 5, v_origin + DOWN * 5]
        square_a = Polygon(*square_a_verts, color=BLUE, fill_opacity=0.5)
        square_a.set_stroke(width=3)

        # Square on side 'b' (length 12) - along the y-axis
        # Vertices: (0,0), (0,12), (-12,12), (-12,0)
        square_b_verts = [v_origin, v_side_b, v_side_b + LEFT * 12, v_origin + LEFT * 12]
        square_b = Polygon(*square_b_verts, color=RED, fill_opacity=0.5)
        square_b.set_stroke(width=3)

        # Square on the hypotenuse (length 13)
        # The hypotenuse connects v_side_a (5,0,0) and v_side_b (0,12,0).
        # The vector along the hypotenuse is v_side_b - v_side_a = (-5, 12, 0).
        # A vector perpendicular to this (rotated 90 degrees anti-clockwise) is (-12, -5, 0).
        # This perpendicular vector has a magnitude of sqrt((-12)^2 + (-5)^2) = sqrt(144 + 25) = sqrt(169) = 13.
        # We use this vector to find the other two vertices of the square, starting from v_side_a and v_side_b.
        perp_vector = np.array([-12, -5, 0])
        square_c_verts = [v_side_a, v_side_b, v_side_b + perp_vector, v_side_a + perp_vector]
        square_c = Polygon(*square_c_verts, color=GREEN, fill_opacity=0.5)
        square_c.set_stroke(width=3)

        # --- Group objects and Position ---
        figure_group = VGroup(triangle, square_a, square_b, square_c)

        # Scale and center the figure to fit well on the screen
        figure_group.scale(0.5) # Adjust scale factor as needed
        figure_group.move_to(ORIGIN)
        figure_group.shift(UP * 0.5) # Shift slightly up

        # --- Add Labels and Explanatory Text ---

        # Labels for triangle side lengths
        label_a = MathTex("5").next_to(square_a, DOWN)
        label_b = MathTex("12").next_to(square_b, LEFT)
        # Position hypotenuse label carefully
        label_c = MathTex("13").next_to(square_c, UP).shift(LEFT * 0.8)

        # Labels for square areas
        area_label_a = MathTex("5^2", "=25").move_to(square_a.get_center())
        area_label_b = MathTex("12^2", "=144").move_to(square_b.get_center())
        area_label_c = MathTex("13^2", "=169").move_to(square_c.get_center())

        # Explanatory text and formula
        theorem_title = Text(
            "Pythagorean Theorem", font_size=48
        ).to_edge(UP)

        explanation_text = Text(
            "The area of the square on the hypotenuse\n"
            "is equal to the sum of the areas\n"
            "of the squares on the other two sides.",
            font_size=24,
        ).next_to(theorem_title, DOWN, buff=0.5)

        # Create the formula steps as separate MathTex objects within a VGroup
        formula_step1 = MathTex("5^2 + 12^2 = 13^2").set_color(YELLOW)
        formula_step2 = MathTex("25 + 144 = 169").set_color(YELLOW)
        formula_step3 = MathTex("169 = 169").set_color(YELLOW)

        # Group them for positioning
        formula_group = VGroup(formula_step1, formula_step2, formula_step3)
        formula_group.arrange(DOWN, aligned_edge=LEFT, buff=0.7) # Arrange steps but we'll only show one at a time

        # Position the first formula step
        formula_step1.next_to(explanation_text, DOWN, buff=0.7)
        # Position subsequent steps at the same location as the first one
        formula_step2.move_to(formula_step1)
        formula_step3.move_to(formula_step1)


        # --- Animate the figure ---

        # Create shapes
        self.play(Create(triangle))
        self.wait(0.5)
        self.play(Create(square_a), Create(square_b))
        self.wait(0.5)
        self.play(Create(square_c))
        self.wait(1)

        # Add side labels
        self.play(Write(label_a), Write(label_b), Write(label_c))
        self.wait(1)

        # Add area labels
        self.play(Write(area_label_a), Write(area_label_b), Write(area_label_c))
        self.wait(1)

        # Show text and formula
        self.play(Write(theorem_title))
        self.wait(0.5)
        self.play(Write(explanation_text))
        self.wait(2)

        # Animate the formula steps using ReplacementTransform
        self.play(Write(formula_step1)) # Write the first step
        self.wait(1)
        # Transform step 1 into step 2
        self.play(ReplacementTransform(formula_step1, formula_step2))
        self.wait(1)
        # Transform step 2 into step 3
        self.play(ReplacementTransform(formula_step2, formula_step3))
        self.wait(2)

        # Optional: Fade everything out at the end
        # Ensure all potentially visible objects are included
        all_objects_to_fade = VGroup(
            triangle, square_a, square_b, square_c,
            label_a, label_b, label_c,
            area_label_a, area_label_b, area_label_c,
            theorem_title, explanation_text,
            formula_step3 # Only the last step is visible here
        )
        self.play(FadeOut(all_objects_to_fade))
        self.wait(1)