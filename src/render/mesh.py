import os
from tqdm import tqdm
from render.shaders import Shaders
import trimesh
import numpy as np
from moderngl import Program, VertexArray


class Mesh:
    """
    Reads all models/textures from the resources/models folder and creates corresponding GPU assets for them.
    (vao, textures) which can then be loaded into a "model" instance using their folder names.
    """
    _instance = None
    mesh_name = ""

    # mesh_name = ""

    @classmethod
    def instance(cls, ctx=None, mesh_name: str = "") -> object:
        """
        Returns the singleton instance of the Mesh class, or creates a new one if it does not already exist.
        :param ctx: Context instance.
        :param mesh_name: Name of the mesh.
        :return: Singleton instance of the Mesh class.
        """
        if cls._instance is None and ctx is not None:
            cls._instance = cls(ctx, mesh_name)
        return cls._instance

    def __init__(self, app, mesh_name: str = "") -> None:
        """
        Constructor.
        :param app: Glw app.
        :param mesh_name: Name of the mesh.
        """
        if Mesh._instance is not None:
            raise RuntimeError("Mesh is a singleton and should not be instantiated more than once")

        self.app = app
        self.data = {}

    def set_mesh_name(self, mesh_name: str) -> None:
        """
        Sets the name of the mesh.
        :param mesh_name: Name of the mesh.
        """
        self.mesh_name = mesh_name

    def get_mesh_name(self) -> str:
        """
        Returns the name of the mesh.
        :return: Name of the mesh.
        """
        return self.mesh_name

    def trimesh_to_vao(self, mesh: trimesh.Trimesh, prog: Program) -> VertexArray:
        """
        Coverts trimesh to vao.
        :param mesh: Current mesh.
        :param prog: Moderngl Program
        :return: Mesh vertex array.
        """
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        indices = mesh.faces.flatten()

        vertex_data = np.hstack((vertices, normals))
        vbo = self.app.ctx.buffer(vertex_data.astype('f4'))
        ibo = self.app.ctx.buffer(indices.astype('i4'))

        vao_content = [
            (vbo, '3f 3f', 'in_position', 'in_normal'),
        ]

        return self.app.ctx.vertex_array(prog, vao_content, ibo)

    def set_data(self) -> None:
        """
        Sets mesh data.
        """
        models_path = os.path.join(os.path.dirname(__file__), '../../resources/models')
        programs = Shaders.instance()
        prog = programs.get('base-flat')

        for root, dirs, files in tqdm(os.walk(models_path), desc="Reading .obj files"):
            # Only reading first file for now, otherwise startup is too slow
            if len(files) > 0:
                len_files = len(files)

                for i in range(len_files):
                    file = files[i]
                    mesh = trimesh.load_mesh(os.path.join(root, file))
                    self.data[file] = (self.trimesh_to_vao(mesh, prog), None)
