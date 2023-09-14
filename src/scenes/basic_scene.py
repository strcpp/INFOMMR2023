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
        pass

    def unload(self) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def render_ui(self) -> None:
        """
        Renders the UI.
        """

    def render(self) -> None:
        """
        Renders all objects in the scene.
        """