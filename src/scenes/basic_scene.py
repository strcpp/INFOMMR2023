from render.model import Model
from render.grid import Grid
from render.skybox import Skybox
from scenes.scene import Scene
from pyrr import Matrix44, Vector3
from light import Light
import imgui
import numpy as np
from typing import List, Optional, Tuple
import os

class BasicScene(Scene):
    """
    Implements the scene of the application.
    """
    models = []
    current_model = ""
    light = None
    skybox = None
    grid = None

    def load(self) -> None:
        self.skybox = Skybox(self.app, skybox='clouds', ext='png')
        self.model = Model(self.app, "m1337.obj")
        self.light = Light(
            position=Vector3([5., 5., 5.], dtype='f4'),
            color=Vector3([1.0, 1.0, 1.0], dtype='f4')
        )
        pass

    def unload(self) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def render_ui(self) -> None:
        """
        Renders the UI.
        """
        imgui.new_frame()

        # Change the style of the entire ImGui interface
        imgui.style_colors_classic()

        # Add an ImGui window
        imgui.begin("Settings", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        imgui.text("Click and drag left/right mouse button to rotate camera.")
        imgui.text("Click and drag middle mouse button to pan camera.")
        imgui.end()
        imgui.render()

        self.app.imgui.render(imgui.get_draw_data())

    def render(self) -> None:
        self.skybox.draw(self.app.camera.projection.matrix, self.app.camera.matrix)
        self.model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )

        self.render_ui()
        """
        Renders all objects in the scene.
        """