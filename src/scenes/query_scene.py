from render.model import Model
from render.skybox import Skybox
from render.mesh import Mesh
from scenes.scene import Scene
from pyrr import Vector3, matrix44
from light import Light
import imgui
import os
from tools.display_statistics import return_neighbors, return_bounding_box, return_shape_descriptors
from tools.save_statistics import save_data
from tqdm import tqdm
from render.lines import Lines
import trimesh
import numpy as np
from tools.descriptor_extraction import *
from numba import njit
from multiprocessing import Pool, cpu_count
from scenes.scene_utils import euclidean_distance, get_bb_lines, get_basis_lines, evaluate_query, get_best_matching_shapes, resample
import pynndescent


def normalize_single_features(mesh_features):
    # Extract the feature descriptors from the dictionary values
    feature_values = [descriptor.get_single_features() for descriptor in mesh_features.values()]

    features_array = np.array(feature_values)

    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0)

    # Avoid division by zero in case of a zero standard deviation
    std[std == 0] = 1

    standardized_features = (features_array - mean) / std

    # Iterate over the items in the dictionary to set the normalized features
    for (key, descriptor), standardized_feature in zip(mesh_features.items(), standardized_features):
        descriptor.normalize_single_features(standardized_feature)


def load_model(path):
    mesh = trimesh.load_mesh(path[0])
    return mesh, path[1], path[2]


class QueryScene(Scene):
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
    current_shading_mode = "flat"
    show_wireframe = False
    selected_class = 0
    selected_model = 0
    original_model_path = os.path.join(os.path.dirname(__file__), '../../resources/models/Default')
    normalized_model_path = os.path.join(os.path.dirname(__file__), '../../resources/models/Normalized')
    average_model = None
    lines = None
    model_bb = []
    model_basis = None
    move_axes_to_barycenter = False
    show_rest_of_ui = False
    neighbor_count = 1
    best_matching_shapes = []
    show_bb = False
    show_axis = False
    all_classes = []
    all_model_names = {}
    all_meshes = {}
    all_descriptors = {}
    all_basis_lines = {}
    all_barycenter_lines = {}
    all_bounding_boxes = {}
    current_descriptor = None
    index = None
    average_vertices = 0
    average_faces = 0
    average_model_class = ""
    distances = None
    show_rest_of_ui_step_5 = False
    show_rest_of_ui_step_6 = False
    shapes_per_class = {}
    evaluate = False
    selected_evaluation_subject = 0
    precisions, recalls, f1_scores = {}, {}, {}
    poorly_sampled = []
    refined = []
    show_poorly_sampled = False
    selected_poorly_sampled = 0
    show_normalized = False

    def load(self) -> None:
        self.skybox = Skybox(self.app, skybox='clouds', ext='png')
        paths_to_load = []

        # Load Normalized models
        for root, dirs, files in os.walk(self.normalized_model_path):
            len_files = len(files)
            if len(files) > 0:
                for i in range(len_files):
                    file = files[i]
                    model_class = os.path.basename(os.path.normpath(root))
                    path = os.path.join(self.normalized_model_path, model_class, file)
                    paths_to_load.append((path, model_class, file))

        # Use multiprocessing to parallelize the loading
        with Pool(processes=cpu_count()) as pool:
            meshes = list(tqdm(pool.imap_unordered(load_model, paths_to_load)))

        for mesh in meshes:
            model_class = mesh[1]
            model_name = mesh[2]

            if model_class not in self.all_classes:
                self.all_classes.append(model_class)
                self.all_model_names[model_class] = []

            self.all_model_names[model_class].append(model_name)
            self.all_meshes[model_name] = mesh[0]

        self.average_model, all_neighbors = return_neighbors()

        # Average model mesh and info
        self.current_model_name = self.average_model["Shape Name"]
        self.current_model = Model(self.app, self.current_model_name, self.all_meshes[self.current_model_name])
        self.current_class = self.average_model["Shape Class"]
        mesh = self.all_meshes[self.current_model_name]
        self.average_vertices = len(mesh.vertices)
        self.average_faces = len(mesh.faces)
        self.average_model_class = self.current_class

        self.all_descriptors = return_shape_descriptors(self.all_model_names, self.all_meshes)
        self.current_descriptor = self.all_descriptors[self.current_model_name]

        self.all_bounding_boxes[self.current_model_name] = get_bb_lines(self.all_meshes[self.current_model_name].bounds)
        self.all_basis_lines[self.current_model_name] = get_basis_lines(self.all_meshes[self.current_model_name].bounds,
                                                                        None)
        self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None,
                                                                             self.all_meshes[
                                                                                 self.current_model_name].centroid)

        # Condition for poorly sampled shapes
        condition = (((all_neighbors['Number of Faces'] < 100) | (all_neighbors['Number of Vertices'] < 100) |
                      (all_neighbors['Number of Faces'] > 50000) | (all_neighbors['Number of Vertices'] > 50000)))
        poorly_sampled = all_neighbors[condition][:1]

        # Resampling poorly sampled outliers
        for _, outlier in tqdm(poorly_sampled.iterrows(), desc="Resampling outliers"):
            name = outlier["Shape Name"]
            model_class = outlier["Shape Class"]
            bounding_box = return_bounding_box(name, None)
            path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Default',
                                outlier['Shape Class'],
                                name)
            mesh = trimesh.load_mesh(path)

            refined_mesh = resample(mesh, self.average_vertices)

            self.refined.append(
                (Model(self.app, name, refined_mesh), get_bb_lines(bounding_box),
                 get_basis_lines(bounding_box, None),
                 get_basis_lines(None, refined_mesh.centroid), name, model_class))

            self.poorly_sampled.append(
                (Model(self.app, name, mesh), get_bb_lines(bounding_box),
                 get_basis_lines(bounding_box, None),
                 get_basis_lines(None, mesh.centroid), name, model_class))

        self.poorly_sampled = sorted(self.poorly_sampled, key=lambda x: x[5])
        self.refined = sorted(self.refined, key=lambda x: x[5])

        self.current_class = self.all_classes[0]
        self.light = Light(
            position=Vector3([5., 5., 5.], dtype='f4'),
            color=Vector3([1.0, 1.0, 1.0], dtype='f4')
        )

        self.current_shading_mode = 0
        self.lines = Lines(self.app, line_width=1)

        normalize_single_features(self.all_descriptors)

        # Setting up ANN query
        print("< Setting up ANN Query...")
        query_shapes = np.array([descriptor.get_normalized_features() for descriptor in self.all_descriptors.values()])

        # Setting all NaN features to 0
        query_shapes = np.nan_to_num(query_shapes, nan=0)

        self.index = pynndescent.NNDescent(query_shapes, n_jobs=-1, random_state=42)
        self.index.prepare()
        print("< Finished setting up ANN Query!")

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

        imgui.set_next_window_position(0, 20, imgui.ONCE)

        # Add an ImGui window
        imgui.begin("Settings", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)

        imgui.text("Click and drag left/right mouse button to rotate camera.")
        imgui.text("Click and drag middle mouse button to pan camera.")

        # General settings
        imgui.text("Visualization Settings: ")
        imgui.spacing()
        imgui.indent(16)
        shading_modes = ["flat", "smooth"]
        clicked, current_item = imgui.combo("Shading Mode", self.current_shading_mode, shading_modes)
        if clicked:
            self.current_shading_mode = current_item
            self.current_model.set_shading(shading_modes[self.current_shading_mode])

        _, self.show_wireframe = imgui.checkbox("Show Wireframe", self.show_wireframe)
        _, self.show_bb = imgui.checkbox("Show Bounding Boxes", self.show_bb)
        _, self.show_axis = imgui.checkbox("Show Axes", self.show_axis)

        if self.show_axis:
            imgui.spacing()
            imgui.indent(16)
            _, self.move_axes_to_barycenter = imgui.checkbox("Move Axes to Barycenter",
                                                             self.move_axes_to_barycenter)
            imgui.unindent(16)

        # Step 2
        imgui.unindent(16)
        imgui.spacing()
        imgui.separator()
        imgui.text("Step 2 - Outlier Resampling: ")
        imgui.spacing()
        imgui.indent(16)
        clicked, self.show_poorly_sampled = imgui.checkbox("Show Poorly Sampled Shapes",
                                                           self.show_poorly_sampled)

        if self.show_poorly_sampled:
            imgui.spacing()
            imgui.indent(16)
            imgui.text("Select Poorly Sampled Shape:")
            _, self.selected_poorly_sampled = imgui.combo(" ",
                                                          self.selected_poorly_sampled,
                                                          [f"{shape[4]}({shape[5]})" for shape in self.poorly_sampled])
            self.show_normalized = False

            self.current_model_name = self.poorly_sampled[self.selected_poorly_sampled][4]
            self.current_class = self.poorly_sampled[self.selected_poorly_sampled][5]
            self.current_model = self.poorly_sampled[self.selected_poorly_sampled][0]
            mesh = self.current_model.mesh

            self.current_descriptor = self.all_descriptors[self.current_model_name]
            self.all_bounding_boxes[self.current_model_name] = get_bb_lines(mesh.bounds)
            self.all_basis_lines[self.current_model_name] = get_basis_lines(mesh.bounds, None)
            self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None, mesh.centroid)

        # Step 3
        imgui.unindent(16)
        if self.show_poorly_sampled:
            imgui.unindent(16)
        imgui.spacing()
        imgui.separator()
        imgui.text("Step 3 - Normalization: ")
        imgui.spacing()
        imgui.indent(16)

        clicked, self.show_normalized = imgui.checkbox("Show Normalized Shapes", self.show_normalized)

        if self.show_normalized:
            imgui.text("Select Normalized Shape:")
            imgui.spacing()
            imgui.indent(16)
            self.show_poorly_sampled = False
            # Add a combo box for classes
            clicked, self.current_class_id = imgui.combo("Classes", self.current_class_id, self.all_classes)
            if clicked:
                self.current_class = self.all_classes[self.current_class_id]

            # Add a combo box for models based on selected class
            if self.current_class and self.current_class in self.all_model_names:
                models_of_current_class = self.all_model_names[self.current_class]
                clicked, self.current_model_id = imgui.combo("Models", self.current_model_id,
                                                             models_of_current_class)
                if clicked:
                    self.current_model_name = models_of_current_class[self.current_model_id]
                    mesh = self.all_meshes[self.current_model_name]
                    self.current_model = Model(self.app, self.current_model_name, mesh)
                    self.current_descriptor = self.all_descriptors[self.current_model_name]
                    self.all_bounding_boxes[self.current_model_name] = get_bb_lines(
                        self.all_meshes[self.current_model_name].bounds)
                    self.all_basis_lines[self.current_model_name] = get_basis_lines(
                        self.all_meshes[self.current_model_name].bounds, None)
                    self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None, mesh.centroid)

        imgui.spacing()
        imgui.separator()
        imgui.text("Query: ")
        imgui.spacing()
        imgui.indent(16)

        _, self.neighbor_count = imgui.input_int("Number of Returned Shapes", self.neighbor_count)

        if imgui.button("Get Best-Matching Shapes (Step 4)"):
            matching_names, self.distances = get_best_matching_shapes(
                self.all_descriptors[self.current_model_name],
                {key: value for key, value in self.all_descriptors.items() if key != self.current_model_name},
                self.neighbor_count)
            self.best_matching_shapes = [(Model(self.app, name, self.all_meshes[name]), name) for name in
                                         matching_names]
            for name in matching_names:
                self.all_bounding_boxes[name] = get_bb_lines(self.all_meshes[name].bounds)
                self.all_basis_lines[name] = get_basis_lines(self.all_meshes[name].bounds, None)
                self.all_barycenter_lines[name] = get_basis_lines(None, self.all_meshes[name].centroid)

        if imgui.button("Get Best-Matching Shapes (Step 5: ANN)"):
            neighbor_indexes, distances = self.index.query(
                np.array([self.all_descriptors[self.current_model_name].get_normalized_features()]),
                k=self.neighbor_count + 1
            )

            self.distances = distances.flatten().tolist()[1:]
            matching_names = [list(self.all_descriptors.keys())[k] for k in neighbor_indexes.flatten().tolist()[1:]]
            self.best_matching_shapes = [(Model(self.app, name, self.all_meshes[name]), name) for name in
                                         matching_names]
            for name in matching_names:
                self.all_bounding_boxes[name] = get_bb_lines(self.all_meshes[name].bounds)
                self.all_basis_lines[name] = get_basis_lines(self.all_meshes[name].bounds, None)
                self.all_barycenter_lines[name] = get_basis_lines(None, self.all_meshes[name].centroid)

        if self.current_model != '':
            if imgui.begin("Descriptors", True):
                imgui.text(f"Shape Class: {self.current_class}")
                imgui.text(f"Number of Vertices: {len(self.all_meshes[self.current_model_name].vertices)}")
                imgui.text(f"Number of Faces: {len(self.all_meshes[self.current_model_name].faces)}")
                imgui.text("{}: {:.2f}".format("Surface area", self.current_descriptor.surface_area))
                imgui.text("{}: {:.2f}".format("Compactness", self.current_descriptor.compactness))
                imgui.text("{}: {:.2f}".format("Rectangularity", self.current_descriptor.rectangularity))
                imgui.text("{}: {:.2f}".format("Diameter", self.current_descriptor.diameter))
                imgui.text("{}: {:.2f}".format("Convexity", self.current_descriptor.convexity))
                imgui.text("{}: {:.2f}".format("Eccentricity", self.current_descriptor.eccentricity))

            # End the window
            imgui.end()

            # Step 6
            imgui.unindent(16)
            imgui.spacing()
            imgui.separator()
            imgui.text("Evaluation: ")
            imgui.spacing()
            imgui.indent(16)
            if imgui.button("Evaluate CBSR System"):
                self.shapes_per_class = {key: len(value) for key, value in self.all_model_names.items()}
                self.show_rest_of_ui_step_5 = False
                self.show_rest_of_ui_step_6 = True

            if self.show_rest_of_ui_step_6:
                _, self.neighbor_count = imgui.input_int("Number of Neighbors", self.neighbor_count)

                if imgui.button("Use Custom Query"):
                    self.precisions, self.recalls, self.f1_scores = evaluate_query(
                        "Custom", self.all_descriptors, self.neighbor_count, self.shapes_per_class,
                        self.all_model_names, None
                    )
                    self.evaluate = True

                if imgui.button("Use ANN"):
                    self.precisions, self.recalls, self.f1_scores = evaluate_query(
                        "ANN", self.all_descriptors, self.neighbor_count, self.shapes_per_class, self.all_model_names,
                        self.index
                    )
                    self.evaluate = True

                if self.evaluate:
                    imgui.set_next_window_position(0, 600, imgui.ONCE)

                    if imgui.begin("Evaluation Results:", True):
                        if self.selected_evaluation_subject == 0:
                            evaluation_subject = "Average"
                        else:
                            evaluation_subject = list(self.shapes_per_class.keys())[
                                self.selected_evaluation_subject - 1]
                        imgui.text(f"{evaluation_subject} Query Precision: {self.precisions[evaluation_subject]:.3f}")
                        imgui.text(f"{evaluation_subject} Query Recall: {self.recalls[evaluation_subject]:.3f}")
                        imgui.text(f"{evaluation_subject} Query F1 Score: {self.f1_scores[evaluation_subject]:.3f}")

                        imgui.spacing()
                        imgui.text("Select Evaluation Subject:")
                        imgui.spacing()
                        imgui.indent(16)
                        changed, self.selected_evaluation_subject = imgui.combo("",
                                                                                self.selected_evaluation_subject,
                                                                                ["Average"] +
                                                                                [shape_name for shape_name
                                                                                 in self.shapes_per_class.keys()])
                    imgui.end()

        imgui.end()
        imgui.render()

        self.app.imgui.render(imgui.get_draw_data())

    def render_shapes(self, color=[1, 1, 1, 1]) -> None:
        def draw_shape(model, name, translation=0):
            model.color = color
            model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )
            model.translate(translation, 0)

            if self.show_bb:
                self.lines.color = [1, 0, 0, 1]
                self.lines.update(self.all_bounding_boxes[name])
                self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix,
                                model.get_model_matrix())
            if self.show_axis:
                self.lines.color = [0, 0, 1, 1]

                if self.move_axes_to_barycenter:
                    self.lines.update(self.all_barycenter_lines[name])
                else:
                    self.lines.update(self.all_basis_lines[name])
                self.app.ctx.depth_func = '1'  # ALWAYS
                self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix,
                                model.get_model_matrix())
                self.app.ctx.depth_func = '<'  # LESS

        if self.current_model != '':
            draw_shape(self.current_model, self.current_model_name)

            translation = 2
            for match in self.best_matching_shapes:
                draw_shape(match[0], match[1], translation)
                translation *= -1
                if translation > 0:
                    translation += 2

    def render(self) -> None:
        """
        Renders all objects in the scene.
        """
        self.skybox.draw(self.app.camera.projection.matrix, self.app.camera.matrix)

        if self.show_wireframe:
            self.app.ctx.wireframe = True
            self.render_shapes(color=[0, 0, 0, 0])
            self.app.ctx.wireframe = False

        self.render_shapes()
        self.render_ui()
