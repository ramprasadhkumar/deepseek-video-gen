
from manim import *

class GeneratedScene(Scene):
    def construct(self):
        from manim import *

        class FallbackScene(Scene):
            def construct(self):
                text = Text("Fallback scene - no code available")
                self.play(Write(text))
                self.wait(2)
