from render.model import Model
from render.skybox import Skybox
from scenes.scene import Scene
from pyrr import Vector3
from light import Light
import imgui
import os
from tools.display_statistics import return_neighbors, return_bounding_box
from  tools.save_statistics import save_refined
from tqdm import tqdm
from moderngl_window.opengl.vao import VAOError
from render.lines import Lines
import moderngl
import trimesh
import numpy as np


def get_bb_lines(bounding_box):
    x0, y0, z0 = bounding_box[0][0], bounding_box[0][1], bounding_box[0][2]
    x1, y1, z1 = bounding_box[1][0], bounding_box[1][1], bounding_box[1][2]

    p000 = (x0, y0, z0)
    p001 = (x0, y0, z1)
    p010 = (x0, y1, z0)
    p011 = (x0, y1, z1)
    p100 = (x1, y0, z0)
    p101 = (x1, y0, z1)
    p110 = (x1, y1, z0)
    p111 = (x1, y1, z1)

    connections = [
        (p000, p001), (p000, p010), (p000, p100),
        (p111, p110), (p111, p101), (p111, p011),
        (p001, p101), (p001, p011),
        (p010, p110), (p010, p011),
        (p100, p101), (p100, p110),
    ]

    return connections

def get_basis_lines(bounding_box):
    # Calculate the model's geometric center
    center_x = (bounding_box[0][0] + bounding_box[1][0]) / 2
    center_y = (bounding_box[0][1] + bounding_box[1][1]) / 2
    center_z = (bounding_box[0][2] + bounding_box[1][2]) / 2
    
    center = (center_x, center_y, center_z)
    
    # Define the offsets for the basis vectors
    offset = 0.25  # This can be adjusted based on desired length of basis vectors
    
    # Calculate endpoints of basis vectors centered at the model's origin
    i_pos = (center_x + offset, center_y, center_z)
    j_pos = (center_x, center_y + offset, center_z)    
    k_pos = (center_x, center_y, center_z + offset)
    
    # Define the lines connecting the center to the endpoints of the basis vectors
    connections = [
        (center, i_pos),
        (center, j_pos),
        (center, k_pos),
    ]
    
    return connections

def refine_mesh(mesh, target_faces_min = 1000, target_faces_max = 50000):
    if len(mesh.faces) == target_faces_min:
        return mesh

    #Subdivide until target reached
    while len(mesh.faces) < target_faces_min:
        mesh = mesh.subdivide()

    #It will probably exceed the target, so decimate it back down
    if len(mesh.faces) > target_faces_max:
        mesh = mesh.simplify_quadratic_decimation(target_faces_max)
    else:
        mesh = mesh.simplify_quadratic_decimation(target_faces_min)

    return mesh

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
    refined_meshes = {}
    outliers = []
    poorly_sampled = []
    selected_poorly_sampled = 0
    lines = None
    model_bb = []
    show_poorly_sampled = False

    def load(self) -> None:
        self.skybox = Skybox(self.app, skybox='clouds', ext='png')
        # Load all models
        
        for root, dirs, files in tqdm(os.walk(self.models_path), desc="Reading .obj files"):
            if(len(files) > 0):
                # How many files to load from each class, usually to speed up devel                
                len_files = len(files) if self.app.config['len_files'] == 0 else self.app.config['len_files']
                for i in range(len_files):
                    file = files[i]
                    model_class = os.path.basename(os.path.normpath(root))
                    if model_class not in self.models:
                        self.models[model_class] = [file]
                    else:
                        self.models[model_class].append(file)

        num_outliers = 10
        # Get average model and outliers
        self.average_model, all_neighbhors = return_neighbors(num_outliers)
        outliers = all_neighbhors[-num_outliers:]
        
        condition = (((all_neighbhors['Number of Faces'] < 100) | (all_neighbhors['Number of Vertices'] < 100) |
                     (all_neighbhors['Number of Faces'] > 50000) | (all_neighbhors['Number of Vertices'] > 50000)))

        poorly_sampled = all_neighbhors[condition]

        for i, p in poorly_sampled.iterrows():
            print(p["Shape Name"])
            print(p["Number of Faces"])
            print(p["Number of Vertices"])
            print("--------------------")
        self.show_poorly_sampled = True
        self.show_wireframe = True
        self.current_model_name = self.average_model["Shape Name"]
        self.current_model = Model(self.app, self.current_model_name)
        self.current_class = self.average_model["Shape Class"]
        self.current_class_id = list(self.models.keys()).index(self.current_class)
        self.current_model_id = self.models[self.current_class].index(self.current_model_name)
        self.lines = Lines(self.app, line_width=1)

        for i, outlier in outliers.iterrows():
            name = outlier["Shape Name"]
            bounding_box = return_bounding_box(name)
            self.outliers.append((Model(self.app, name), get_bb_lines(bounding_box), get_basis_lines(bounding_box)))

        for i, outlier in poorly_sampled.iterrows():
            name = outlier["Shape Name"]
            bounding_box = return_bounding_box(name)
            path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', outlier['Shape Class'], name)
            mesh = trimesh.load_mesh(path)

            print(name)
            print(f"Initial face count: {len(mesh.faces)} | Initial vertex count: {len(mesh.vertices)}")
            refined_mesh = refine_mesh(mesh)
            self.refined_meshes[name] = (name, outlier['Shape Class'], refined_mesh)
            print(f"Refined face count: {len(refined_mesh.faces)} | Refined vertex count: {len(refined_mesh.vertices)}")
            print("------------------------")

            self.poorly_sampled.append((Model(self.app, name, refined_mesh), get_bb_lines(bounding_box), get_basis_lines(bounding_box), name))

        for i, models in self.models.items():
            for model_name in models:
                if model_name not in self.refined_meshes:
                    bounding_box = return_bounding_box(model_name)
                    self.poorly_sampled.append((Model(self.app, model_name, None), get_bb_lines(bounding_box), get_basis_lines(bounding_box), model_name))


        self.light = Light(
            position=Vector3([5., 5., 5.], dtype='f4'),
            color=Vector3([1.0, 1.0, 1.0], dtype='f4')
        )

        self.current_shading_mode = 0
        bounding_box = return_bounding_box(self.current_model_name)
        self.model_bb = get_bb_lines(bounding_box)
        self.model_basis = get_basis_lines(bounding_box)


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
        clicked, self.show_poorly_sampled = imgui.checkbox("Show Poorly Sampled", self.show_poorly_sampled)
        if self.show_poorly_sampled:
            _, self.selected_poorly_sampled = imgui.combo("Poorly Sampled Shapes", 
                                                    self.selected_poorly_sampled, [shape[3] for shape in self.poorly_sampled])


        if imgui.button("Save Refined Statistics"):
            for model_class, model_name in self.models.items():
                for name in model_name:
                    if name not in self.refined_meshes:
                        path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', model_class, name)
                        self.refined_meshes[name] = (model_class, name, trimesh.load_mesh(path))

            # print(self.refined_meshes)
            save_refined(self.refined_meshes)
            pass

        imgui.end()
        imgui.render()

        self.app.imgui.render(imgui.get_draw_data())

    def render(self) -> None:
        """
        Renders all objects in the scene.
        """
        self.skybox.draw(self.app.camera.projection.matrix, self.app.camera.matrix)
        if self.show_wireframe:
            self.app.ctx.wireframe = True

            if not self.show_poorly_sampled:
                self.current_model.color = [0, 0, 0]
                self.current_model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )

                translation = 1
                for outlier in self.outliers:
                    outlier, bounds, = outlier[0],outlier[1]
                    outlier.color = [1, 1, 1]
                    try:
                        outlier.draw(
                            self.app.camera.projection.matrix,
                            self.app.camera.matrix,
                            self.light
                        )
                        outlier.translate(translation, 0)
                        translation *= -1
                        if translation > 0:
                            translation += 1
                    except VAOError:
                        pass
            else:
                model = self.poorly_sampled[self.selected_poorly_sampled][0]
                model.color = [0, 0, 0]
                model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )

            self.app.ctx.wireframe = False

        if not self.show_poorly_sampled:
            self.current_model.color = [1, 1, 1]
            self.current_model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )

            self.lines.update(self.model_bb)
            self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix, self.current_model.get_model_matrix())
            self.lines.color = [0,0,1,1]
            self.lines.update(self.model_basis)
            self.app.ctx.depth_func = '1' # ALWAYS 
            self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix, self.current_model.get_model_matrix())
            self.app.ctx.depth_func = '<' # LESS
            self.lines.color = [1,0,0,1]
            
            translation = 1
            for outlier in self.outliers:
                outlier, bounds, basis = outlier[0],outlier[1], outlier[2]
                outlier.color = [1, 1, 1]
                try:
                    outlier.draw(
                        self.app.camera.projection.matrix,
                        self.app.camera.matrix,
                        self.light
                    )
                    outlier.translate(translation, 0)
                    translation *= -1
                    if translation > 0:
                        translation += 1
                        
                    self.lines.update(bounds)
                    self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix, outlier.get_model_matrix())
                    self.lines.color = [0,0,1,1]
                    self.app.ctx.depth_func = '1' # ALWAYS 
                    self.lines.update(basis)
                    self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix, outlier.get_model_matrix())
                    self.app.ctx.depth_func = '<' # LESS
                    self.lines.color = [1,0,0,1]

                except VAOError:
                    pass
        else:
            sample = self.poorly_sampled[self.selected_poorly_sampled] 
            model = sample[0]
            model.color = [1, 1, 1]
            model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )

            self.lines.update(sample[1])
            self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix, model.get_model_matrix())
            self.lines.color = [0,0,1,1]
            self.lines.update(sample[2])
            self.app.ctx.depth_func = '1' # ALWAYS 
            self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix, model.get_model_matrix())
            self.app.ctx.depth_func = '<' # LESS
            self.lines.color = [1,0,0,1]
            

        self.render_ui()

