import os
import time
from tqdm import tqdm
from render.shaders import Shaders


class Mesh:
    """
    Reads all models/textures from the resources/models folder and creates corresponding GPU assets for them.
    (vao, textures) which can then be loaded into a "model" instance using their folder names.
    """
    _instance = None
    mesh_name = ""

    @classmethod
    def instance(cls, ctx=None, mesh_name="") -> object:
        """
        Returns the singleton instance of the Mesh class, or creates a new one if it does not already exist.
        :param ctx: Context instance.
        :return: Singleton instance of the Mesh class.
        """
        if cls._instance is None and ctx is not None:
            print(mesh_name)
            cls._instance = cls(ctx, mesh_name)
        return cls._instance

    def __init__(self, app, mesh_name="") -> None:
        """
        Constructor.
        :param app: Glw app.
        """
        if Mesh._instance is not None:
            raise RuntimeError("Mesh is a singleton and should not be instantiated more than once")

        self.app = app
        self.data = {}

        self.mesh_name = mesh_name

        start = time.time()
        models_path = os.path.join(os.path.dirname(__file__), '../../resources/models')

        for root, dirs, files in os.walk(models_path):
            if len(files) > 0:
                if self.mesh_name == "":
                    filename = files[0]
                    obj = self.app.load_scene(os.path.join(root, filename))
                    self.data[filename] = (obj.root_nodes[0].mesh.vao, None)
                    break
                else:
                    for file in files:
                        if file == self.mesh_name:
                            obj = self.app.load_scene(os.path.join(root, file))
                            self.data[file] = (obj.root_nodes[0].mesh.vao, None)
                            break

        end = time.time()

        print("elapsed??: ", end - start)

    def set_mesh_name(self, mesh_name):
        self.mesh_name = mesh_name

    def get_mesh_name(self):
        return self.mesh_name

    def set_data(self):
        models_path = os.path.join(os.path.dirname(__file__), '../../resources/models')

        for root, dirs, files in os.walk(models_path):
            if len(files) > 0:
                for file in files:
                    if file == self.mesh_name:
                        obj = self.app.load_scene(os.path.join(root, file))
                        self.data[file] = (obj.root_nodes[0].mesh.vao, None)
                        break
