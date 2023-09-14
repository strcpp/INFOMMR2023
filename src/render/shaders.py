from moderngl import Program


class Shaders:
    """
   Loads/stores all shaders to be used by the program.
    """
    _instance = None

    @classmethod
    def instance(cls, app=None) -> object:
        """
        Returns the singleton instance of the Shaders class, or creates a new one if it does not already exist.
        :param app: Glw app.
        :return: Singleton instance of the Shaders class.
        """
        if cls._instance is None and app is not None:
            cls._instance = cls(app)
        return cls._instance

    def __init__(self, app) -> None:
        """
        Constructor.
        :param app: Main app.
        """
        if Shaders._instance is not None:
            raise RuntimeError("Shaders is a singleton and should not be instantiated more than once")
        self.shaders = {}
        self.app = app
        self.shaders['base-smooth'] = self.app.load_program("shaders/base-smooth.glsl")
        self.shaders['base-flat'] = self.app.load_program("shaders/base-flat.glsl")
        
        self.shaders['skybox'] = self.app.load_program("shaders/skybox.glsl")
        self.shaders['grid'] = self.app.load_program("shaders/grid.glsl")

    def get(self, name: str) -> Program:
        """
        Returns the shader.
        :param name: Shader name.
        :return: Current shader.
        """
        return self.shaders[name]

    def destroy(self) -> None:
        """
        Destroys the shader.
        """
        [shader.release() for shader in self.shaders.values()]
