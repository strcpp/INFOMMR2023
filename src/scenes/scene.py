from abc import abstractmethod
from render.model import Model

from typing import Optional


class Entity:
    """
    Represents an entity in the scene.
    """

    def __init__(self, name: str, model: Model):
        """
        Constructor.
        :param name: Entity name.
        :param model: The model associated with the entity.
        """
        self.name = name
        self.model = model


class Scene:
    """
    Represents a scene in the application.
    """

    def __init__(self, app) -> None:
        """
        Constructor.
        :param app: Glw app.
        """
        self.current_animation_names = None
        self.current_model_entity = None
        self.current_model = ""
        self.model_names_in_scene = []
        self.app = app
        self.entities = []
        self.model_counter = 0

    def add_entity(self, name: str, model: Model) -> None:
        """
        Adds an entity to the scene.
        :param name: Entity name.
        :param model: The model associated with the entity.
        """
        self.entities.append(Entity(name, model))

    def add_model(self, name: str) -> str:
        """
        Adds a model to the list of models that are currently being rendered.
        :param name: Model name.
        :return: Unique model name identifier.
        """
        self.model_counter += 1
        unique_name = f'{str(self.model_counter)} - {name}'
        self.add_entity(unique_name, Model(self.app, name))
        self.model_names_in_scene.append(unique_name)

        return unique_name

    def find(self, name: str) -> Optional[Model]:
        """
        Finds a model in the scene based on its name.
        :param name: Name of the model to find.
        :return: The found model, or None if not found.
        """
        for entity in self.entities:
            if entity.name == name:
                return entity.model
        return None

    def set_model(self, model_name: str) -> None:
        """
        Initializes the current active model, associates it with an entity and retrieves its animations.
        :param model_name: Name of the model to initialize.
        """
        self.current_model = model_name
        self.current_model_entity = self.find(self.current_model)
        self.current_animation_names = list(map(lambda a: a.name, self.current_model_entity.animations))

    @abstractmethod
    def load(self) -> None:
        """
        Abstract method for loading the scene.
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Abstract method for unloading the scene.
        """
        pass

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Abstract method for updating the scene.
        :param dt: Update time step.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Abstract method for rendering the scene.
        """
        pass

    @abstractmethod
    def key_event(self, key: int, action: str) -> None:
        """
        Abstract method for key events.
        :param key: Key code or identifier associated with the key event.
        :param action: Action performed on the key (e.g., "press", "release").
        """
        pass
