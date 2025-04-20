# Import necessary libraries
from manim import *
from manim.utils import *
from manim.scene.scene import *
from manim.scene.camera import *
from manim.scene.types import *
from manim.mobject.standards import *

# Create a right triangle with sides 5, 12, 13
def create_triangle():
    right_triangle = RightTriangle(
        right_length=13,
        right_angle=30,
        left_length=5,
        left_angle=60,
    )
    return right_triangle

# Create squares on each side
def create_squares():
    # Square on the right side
    right_square = Square(
        side_length=13,
        color=BLUE,
        fill_color=BLUE,
    )
    # Square on the left side
    left_square = Square(
        side_length=5,
        color=RED,
        fill_color=RED,
    )
    # Square on the left hypotenuse
    hypotenuse_square = Square(
        side_length=12,
        color=GREEN,
        fill_color=GREEN,
    )
    return right_square, left_square, hypotenus_square

# Create the main scene
def create_scene():
    # Set up the camera
    camera = Camera(
        look_at=(0,0,0),
        type="perspective"
    )
    # Create the triangle and squares
    triangle = create_triangle()
    right_sq, left_sq, hyp_sq = create_squares()
    # Add triangle and squares to the scene
    scene = Scene(
        camera=camera,
        x_axis=AXIS,
        y_axis=AXIS,
        background=WHITE
    )
    scene.add(triangle)
    scene.add(right_sq)
    scene.add(left_sq)
    scene.add(hyp_sq)
    return scene

# Add labels to the triangle
def add_labels(triangle):
    # Add labels to the sides
    triangle.add_label(
        position=triangle.left_angle,
        label="60^\circ",
        color=BLACK
    )
    triangle.add_label(
        position=triangle.right_angle,
        label="30^\circ",
        color=BLACK
    )
    triangle.add_label(
        position=triangle.hypotenuse,
        label="13",
        color=BLACK
    )
    triangle.add_label(
        position=triangle.left_side,
        label="5",
        color=BLACK
    )
    triangle.add_label(
        position=triangle.right_side,
        label="12",
        color=BLACK
    )
    return triangle

# Create the entire manim document
def main():
    scene = create_scene()
    add_labels(scene)
    # Use a loop to animate the squares
    # This will move the squares to form the Pythagorean theorem demonstration
    # Note: The animation might need to be adjusted for better visual effect
    for i in range(20):
        scene.animate()
        print(f"Step {i}: Animate and move squares")
    # Use a loop to show each step of the proof
    for i in range(20):
        scene.animate()
        print(f"Step {i}: Show step of the proof")
    # Finally, play the animation
    # scene.play()
    # scene.animate()
    # scene.finish()

# Run the manim code
if __name__ == "__main__":
    main()