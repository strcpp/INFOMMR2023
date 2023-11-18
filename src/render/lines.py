import numpy as np
from pyrr import Matrix44
from render.shaders import Shaders
import moderngl
from typing import List, Tuple

MAX_LINE_BUFFER_SIZE = 5000


def build_lines(lines: List[Tuple[Matrix44, Matrix44]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a line mesh from a list of line segments defined by their start and end points.
    :param lines: List of tuples, where each tuple contains the start and end points of a line segment.
    :returns: Tuple containing the vertex data and index data of the line mesh.
    """
    vertices = []
    indices = []
    index_counter = 0

    for line in lines:
        start, end = line
        vertices.extend(start)
        vertices.extend(end)

        indices.append(index_counter)
        indices.append(index_counter + 1)

        index_counter += 2

    vertex_data = np.array(vertices, dtype=np.float32)
    index_data = np.array(indices, dtype=np.uint32)

    return vertex_data, index_data


class Lines:
    """
    Implements a model's skeleton as lines.
    """

    def __init__(self, app, line_width: int = 1, color: list[int] | None = None, lines: list | None = None) -> None:
        """
        Constructor.
        :param app: Glw app.
        :param line_width: Width of lines.
        :param color: Color of lines.
        :param lines: List of already drawn lines.
        """
        if lines is None:
            lines = []
        if color is None:
            color = [1, 0, 0, 1]
        self.app = app
        self.lineWidth = line_width
        self.color = color
        programs = Shaders.instance()
        self.line_prog = programs.get('lines')
        self.lines = lines

        vertices, indices = build_lines(lines)

        self.vbo = self.app.ctx.buffer(reserve=MAX_LINE_BUFFER_SIZE, dynamic=True)
        self.ibo = self.app.ctx.buffer(reserve=MAX_LINE_BUFFER_SIZE, dynamic=True)

        self.vbo.write(vertices)
        self.ibo.write(indices)

        self.vao = self.app.ctx.simple_vertex_array(self.line_prog, self.vbo, "position", index_buffer=self.ibo)

    def update(self, lines: List[Tuple[Matrix44, Matrix44]]) -> None:
        """
        Update method.
        :param lines: List of already drawn lines.
        """
        vertices, indices = build_lines(lines)
        self.vbo.clear()
        self.ibo.clear()
        self.vbo.write(vertices)
        self.ibo.write(indices)

    def draw(self, proj_matrix: Matrix44, view_matrix: Matrix44, model_matrix: np.array) -> None:
        """
        Draws the skeleton lines.
        :param proj_matrix: Projection matrix.
        :param view_matrix: View matrix.
        :param model_matrix: Transformation matrix.
        """
        self.line_prog["img_width"].value = self.app.window_size[0]
        self.line_prog["img_height"].value = self.app.window_size[1]
        self.line_prog["line_thickness"].value = self.lineWidth

        self.line_prog['model'].write(model_matrix)
        self.line_prog['view'].write(view_matrix)
        self.line_prog['projection'].write(proj_matrix)
        self.line_prog['color'].value = self.color

        self.vao.render(moderngl.LINES)