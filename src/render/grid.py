import numpy as np
from pyrr import Quaternion, Vector3, Matrix44
from render.shaders import Shaders

from typing import Optional


class Grid:
    """
    Implements a grid plane.
    """
    def __init__(self, app, color: Optional[np.ndarray] = None, size: int = 200) -> None:
        """
        Constructor.
        :param app: Glw app.
        :param color: Grid color.
        :param size: Grid size.
        """
        self.app = app
        self.color = color
        programs = Shaders.instance()
        self.prog = programs.get('grid')

        vbo = self.app.ctx.buffer(np.array([
            # Vertices          # Texture Coordinates
            [-0.5, 0.5, 0.0, 0.0, 1.0],  # Top-left vertex
            [-0.5, -0.5, 0.0, 0.0, 0.0],  # Bottom-left vertex
            [0.5, -0.5, 0.0, 1.0, 0.0],  # Bottom-right vertex
            [0.5, 0.5, 0.0, 1.0, 1.0],  # Top-right vertex
        ], dtype=np.float32))

        ibo = self.app.ctx.buffer(np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32))

        self.vao = self.app.ctx.vertex_array(self.prog, [(vbo, '3f 2f', 'position', 'in_texcoord_0')], ibo)

        self.translation = Vector3([0.0, 0.0, 0.0])
        self.rotation = Quaternion.from_x_rotation(np.pi / 2.0)
        self.scale = Vector3([size, size, size])

    def get_model_matrix(self) -> np.ndarray:
        """
        Returns the matrix of the model.
        :return: Model matrix.
        """
        trans = Matrix44.from_translation(self.translation)
        rot = Matrix44.from_quaternion(self.rotation)
        scale = Matrix44.from_scale(self.scale)
        model = trans * rot * scale

        return np.array(model, dtype='f4')

    def draw(self, proj_matrix: Matrix44, camera) -> None:
        """
        Draws a grid plane.
        :param proj_matrix: Projection matrix.
        :param camera: Application camera.
        """
        self.prog['model'].write(self.get_model_matrix())
        self.prog['view'].write(camera.matrix)
        self.prog['projection'].write(proj_matrix)
        self.prog['cameraPos'].write(np.array(camera.position, dtype='f4'))

        self.vao.render()
