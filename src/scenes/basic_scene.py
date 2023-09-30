from render.model import Model
from render.skybox import Skybox
from scenes.scene import Scene
from pyrr import Vector3
from light import Light
import imgui
import os
from tools.display_statistics import return_neighbors, return_bounding_box
from tqdm import tqdm
from moderngl_window.opengl.vao import VAOError


class BasicScene(Scene):
    """
    Implements the scene of the application.
    """
    models = {}
    current_model = None
    current_model_name = ""
    current_model_id = 0
    current_class_id = 0
    current_class = ""
    light = None
    skybox = None
    grid = None
    current_shading_mode = "flat"
    show_wireframe = False
    selected_class = 0
    selected_model = 0
    models_path = os.path.join(os.path.dirname(__file__), '../../resources/models')
    average_model = None
    outliers = []

    def load(self) -> None:
        self.skybox = Skybox(self.app, skybox='clouds', ext='png')

        # Load all models
        for root, dirs, files in tqdm(os.walk(self.models_path), desc="Reading .obj files"):
            for file in files:
                model_class = os.path.basename(os.path.normpath(root))
                if model_class not in self.models:
                    self.models[model_class] = [file]
                else:
                    self.models[model_class].append(file)

        # Get average model and 5 outliers
        self.average_model, outliers = return_neighbors(2)

        self.current_model_name = self.average_model["Shape Name"]
        self.current_model = Model(self.app, self.current_model_name)
        self.current_class = self.average_model["Shape Class"]
        self.current_class_id = list(self.models.keys()).index(self.current_class)
        self.current_model_id = self.models[self.current_class].index(self.current_model_name)

        for outlier in outliers["Shape Name"]:
            self.outliers.append(Model(self.app, outlier))

        self.light = Light(
            position=Vector3([5., 5., 5.], dtype='f4'),
            color=Vector3([1.0, 1.0, 1.0], dtype='f4')
        )

        self.current_shading_mode = 0

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

        imgui.text("Select a class")
        _, selected_class = imgui.combo("##class_combo", self.current_class_id, list(self.models.keys()))
        if selected_class != -1 and selected_class != self.current_class_id:
            self.current_class_id = selected_class
            self.current_model_id = 0
            self.current_model_name = list(self.models.items())[self.current_class_id][1][0]
            self.current_model = Model(self.app, self.current_model_name)
            self.current_class = list(self.models.items())[self.current_class_id][0]

        imgui.text("Select a model")
        _, selected_model = imgui.combo("##model_combo", self.current_model_id,
                                        list(self.models.items())[self.current_class_id][1][:])
        if selected_model != -1 and selected_model != self.current_model_id:
            self.current_model_id = selected_model
            self.current_model_name = list(self.models.items())[self.current_class_id][1][self.current_model_id]
            self.current_model = Model(self.app, self.current_model_name)
            self.current_class = list(self.models.items())[self.current_class_id][0]

        # Begin the combo
        shading_modes = ["flat", "smooth"]
        clicked, current_item = imgui.combo("Shading Mode", self.current_shading_mode, shading_modes)
        if clicked:
            self.current_shading_mode = current_item
            self.current_model.set_shading(shading_modes[self.current_shading_mode])

        clicked, self.show_wireframe = imgui.checkbox("Show Wireframe", self.show_wireframe)

        imgui.end()
        imgui.render()

        self.app.imgui.render(imgui.get_draw_data())

    def render(self) -> None:
        self.skybox.draw(self.app.camera.projection.matrix, self.app.camera.matrix)
        if self.show_wireframe:
            self.app.ctx.wireframe = True

            self.current_model.color = [0, 0, 0]
            self.current_model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )

            for outlier in self.outliers:
                outlier.color = [1, 1, 1]
                try:
                    outlier.draw(
                        self.app.camera.projection.matrix,
                        self.app.camera.matrix,
                        self.light
                    )
                except VAOError:
                    pass

            self.app.ctx.wireframe = False

        bounding_box = return_bounding_box(self.current_model_name)
        # Not sure how to draw it

        self.current_model.color = [1, 1, 1]

        self.current_model.draw(
            self.app.camera.projection.matrix,
            self.app.camera.matrix,
            self.light
        )

        for outlier in self.outliers:
            outlier.color = [1, 1, 1]
            try:
                outlier.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )
            except VAOError:
                pass

        self.render_ui()
        """
        Renders all objects in the scene.
        """
