from render.model import Model
from render.skybox import Skybox
from scenes.scene import Scene
from pyrr import Vector3
from light import Light
import imgui
import os
from tools.display_statistics import return_neighbors, return_bounding_box
from tools.save_statistics import save_data
from tqdm import tqdm
from render.lines import Lines
import trimesh
import numpy as np
from tools.descriptor_extraction import *
import pynndescent
from scenes.scene_utils import *
from render.mesh import Mesh

class BasicScene(Scene):
    """
    Implements the scene of the application.
    """
    models = {}
    current_model = None
    current_model_name = ""
    current_class = ""
    light = None
    skybox = None
    grid = None
    current_shading_mode = "flat"
    show_wireframe = True
    selected_class = 0
    selected_model = 0
    models_path = os.path.join(os.path.dirname(__file__), '../../resources/Default/models')
    average_model = None
    refined_meshes = {}
    poorly_sampled = []
    refined = []
    selected_poorly_sampled = 0
    lines = None
    model_bb = []
    show_poorly_sampled = False
    normalized = []
    model_basis = None
    show_normalized = False
    selected_normalized = 0
    move_axes_to_barycenter = False
    show_rest_of_ui_step_5 = False
    show_rest_of_ui_step_6 = False
    neighbor_count = 1
    best_matching_shapes = []
    show_bb = False
    show_axis = False
    target_faces = 0
    average_vertices = 0
    average_faces = 0
    average_model_class = ""
    distances = []
    index = None
    shapes_per_class = None
    evaluate = False
    precisions = 0
    recalls = 0
    f1_scores = 0
    selected_evaluation_subject = 0

    def load(self) -> None:
        Mesh.instance(self.app).set_data()
        self.skybox = Skybox(self.app, skybox='clouds', ext='png')

        # Load all models
        for root, dirs, files in os.walk(self.models_path):
            if len(files) > 0:
                # How many files to load from each class, usually to speed up devel                
                len_files = len(files) if self.app.config['len_files'] == 0 else self.app.config['len_files']
                for i in range(len_files):
                    file = files[i]
                    model_class = os.path.basename(os.path.normpath(root))
                    if model_class not in self.models:
                        self.models[model_class] = [file]
                    else:
                        self.models[model_class].append(file)

        # Get average model and outliers
        self.average_model, all_neighbors = return_neighbors()

        # Condition for poorly sampled shapes
        condition = (((all_neighbors['Number of Faces'] < 100) | (all_neighbors['Number of Vertices'] < 100) |
                      (all_neighbors['Number of Faces'] > 50000) | (all_neighbors['Number of Vertices'] > 50000)))

        poorly_sampled = all_neighbors[condition]

        # Average model mesh and info
        self.current_model_name = self.average_model["Shape Name"]
        self.current_model = Model(self.app, self.current_model_name)
        self.current_class = self.average_model["Shape Class"]
        self.lines = Lines(self.app, line_width=1)
        self.average_model = self.current_model
        path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Default', self.current_class,
                            self.current_model_name)
        mesh = trimesh.load_mesh(path)
        self.average_vertices = len(mesh.vertices)
        self.average_faces = len(mesh.faces)
        self.target_faces = len(mesh.vertices)
        self.average_model_class = self.current_class

        # Resampling poorly sampled outliers
        for i, outlier in tqdm(poorly_sampled.iterrows(), desc="Resampling outliers"):
            name = outlier["Shape Name"]
            model_class = outlier["Shape Class"]
            bounding_box = return_bounding_box(name, None)
            path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources',  'models', 'Default',  outlier['Shape Class'],
                                name)
            mesh = trimesh.load_mesh(path)
            refined_mesh = resample(mesh, self.target_faces)
            self.refined_meshes[name] = (name, outlier['Shape Class'], refined_mesh)

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

        # Normalizing all shapes after resampling
        self.normalized = self.refined.copy()
        for model_class, models in self.models.items():
            for model_name in models:
                if model_name not in self.refined_meshes:
                    bounding_box = return_bounding_box(model_name, None)
                    self.normalized.append((Model(self.app, model_name, None), get_bb_lines(bounding_box),
                                            get_basis_lines(bounding_box, None), None, model_name, model_class))

        for i, model in tqdm(enumerate(self.normalized), desc="Normalizing shapes"):
            model_class = model[-1]
            name = model[-2]
            path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Default',  model_class, name)
            mesh = trimesh.load_mesh(path)

            # Step 1: Resample
            mesh.process()
            mesh.remove_duplicate_faces()
            mesh = resample(mesh, self.target_faces)

            # Step 2: Translation
            barycenter = mesh.centroid
            mesh.apply_translation(-barycenter)

            # Step 3: Pose (alignment)
            covariance_matrix = np.cov(np.transpose(mesh.vertices))
            eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
            sorted_indices = np.argsort(eig_values)[::-1]
            major = eig_vectors[:, sorted_indices[0]]
            medium = eig_vectors[:, sorted_indices[1]]
            minor = np.cross(major, medium)

            dot_product = np.dot(mesh.vertices, np.array([major, medium, minor]))

            mesh = trimesh.Trimesh(dot_product, mesh.faces)

            # Step 4: Orientation
            f = np.sum(np.sign(mesh.triangles_center) * (np.square(mesh.triangles_center)))
            mesh = trimesh.Trimesh((mesh.vertices * np.sign(f)), mesh.faces)

            # Step 5: Size
            max_dimension = max(mesh.extents)
            scale_factor = 1.0 / max_dimension
            mesh.apply_scale(scale_factor)
            bounding_box = return_bounding_box(None, mesh)
            descriptors = ShapeDescriptors.from_mesh(mesh, model_class, name)

            self.normalized[i] = (
                Model(self.app, name, mesh), get_bb_lines(bounding_box), get_basis_lines(bounding_box, None),
                get_basis_lines(None, mesh.centroid), name, model_class, descriptors)

        self.normalized = sorted(self.normalized, key=lambda x: x[5])\

        # Normalizing all single features
        normalize_single_features([element[6] for element in self.normalized])

        # Setting up ANN query
        print("< Setting up ANN Query...")
        query_shapes = np.array([shape[6].get_normalized_features() for shape in self.normalized])

        # Setting all NaN features to 0
        query_shapes = np.nan_to_num(query_shapes, nan=0)

        self.index = pynndescent.NNDescent(query_shapes, n_jobs=-1, random_state=42)
        self.index.prepare()
        print("< Finished setting up ANN Query!")

        self.light = Light(
            position=Vector3([5., 5., 5.], dtype='f4'),
            color=Vector3([1.0, 1.0, 1.0], dtype='f4')
        )

        self.current_shading_mode = 0
        bounding_box = return_bounding_box(self.current_model_name, None)

        self.model_bb = get_bb_lines(bounding_box)
        self.model_basis = get_basis_lines(bounding_box, None)

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

        clicked, self.show_wireframe = imgui.checkbox("Show Wireframe", self.show_wireframe)
        clicked, self.show_bb = imgui.checkbox("Show Bounding Boxes", self.show_bb)
        clicked, self.show_axis = imgui.checkbox("Show Axes", self.show_axis)

        clicked, self.move_axes_to_barycenter = imgui.checkbox("Move Axes to Barycenter",
                                                               self.move_axes_to_barycenter)

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
                                                          [f"{shape[4]}({shape[5]}" for shape in self.poorly_sampled])
            self.show_normalized = False

            if imgui.button("Save Refined Statistics"):
                for model_class, model_name in self.models.items():
                    for name in model_name:
                        if name not in self.refined_meshes:
                            path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources','models',  'Default',
                                                model_class,
                                                name)
                            self.refined_meshes[name] = (model_class, name, trimesh.load_mesh(path))
                save_data(self.refined_meshes)
                pass

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
            changed, self.selected_normalized = imgui.combo(" ",
                                                            self.selected_normalized,
                                                            [f"{shape[4]}({shape[5]})" for shape in self.normalized])
            self.show_poorly_sampled = False
            if changed:
                self.best_matching_shapes = []

        imgui.set_next_window_position(420, 20, imgui.ONCE)
        if imgui.begin("Info", True):
            if self.show_poorly_sampled:
                imgui.columns(2, None, True)
                imgui.set_column_width(-1, 200)
                sample = self.poorly_sampled[self.selected_poorly_sampled]
                imgui.text(f"Model Class: {sample[5]}")
                imgui.text(f"Number of Vertices: {len(sample[0].mesh.vertices)}")
                imgui.text(f"Number of Faces: {len(sample[0].mesh.faces)}")
                imgui.text("")
                imgui.next_column()
                imgui.set_column_width(-1, 180)
                sample = self.refined[self.selected_poorly_sampled]
                imgui.text(f"Model Class: {sample[5]}")
                imgui.text(f"Number of Vertices: {len(sample[0].mesh.vertices)}")
                imgui.text(f"Number of Faces: {len(sample[0].mesh.faces)}")
            elif self.show_normalized:
                imgui.columns(4, None, True)
                aligned_shapes = [self.normalized[self.selected_normalized]]
                aligned_distances = [0]
                for i in range(0, len(self.best_matching_shapes), 2):
                    try:
                        shape2 = self.best_matching_shapes[i + 1]
                        aligned_shapes.insert(0, shape2)
                        aligned_distances.insert(0, self.distances[i + 1])
                    except IndexError:
                        pass
                    shape1 = self.best_matching_shapes[i]
                    aligned_shapes.insert(len(aligned_shapes), shape1)
                    aligned_distances.insert(len(aligned_shapes), self.distances[i])

                for i, shape in enumerate(aligned_shapes):
                    imgui.set_column_width(-1, 230)
                    descriptors = shape[6]
                    imgui.text(f"Model Class: {shape[5]}")
                    imgui.text(f"Distance: {aligned_distances[i]:.3f}")
                    imgui.text(f"Number of Vertices: {descriptors.n_vertices}")
                    imgui.text(f"Number of Faces: {descriptors.n_faces}")
                    imgui.text("{}: {:.2f}".format("Surface area", descriptors.surface_area))
                    imgui.text("{}: {:.2f}".format("Compactness", descriptors.compactness))
                    imgui.text("{}: {:.2f}".format("Rectangularity", descriptors.rectangularity))
                    imgui.text("{}: {:.2f}".format("Diameter", descriptors.diameter))
                    imgui.text("{}: {:.2f}".format("Convexity", descriptors.convexity))
                    imgui.text("{}: {:.2f}".format("Eccentricity", descriptors.eccentricity))

                    if imgui.button("Save Distributions"):
                        descriptors.save_A3_histogram_image()
                        descriptors.save_D1_histogram_image()
                        descriptors.save_D2_histogram_image()
                        descriptors.save_D3_histogram_image()
                        descriptors.save_D4_histogram_image()
                    imgui.text("")

                    imgui.next_column()
            else:
                imgui.text(f"Model Class: {self.average_model_class}")
                imgui.text(f"Number of Vertices: {self.average_vertices}")
                imgui.text(f"Number of Faces: {self.average_faces}")

            imgui.end()

        # Steps 4 and 5
        imgui.unindent(16)
        if self.show_normalized or self.selected_poorly_sampled:
            imgui.unindent(16)
        imgui.spacing()
        imgui.separator()
        imgui.text("Query: ")
        imgui.spacing()
        imgui.indent(16)
        if self.show_normalized:
            if imgui.button("Compute Distance to all Shapes"):
                self.show_rest_of_ui_step_5 = True
                self.show_rest_of_ui_step_6 = False

        if self.show_rest_of_ui_step_5:
            _, self.neighbor_count = imgui.input_int("Number of Returned Shapes", self.neighbor_count)

            if imgui.button("Get Best-Matching Shapes (Step 4)"):
                self.best_matching_shapes, self.distances = get_best_matching_shapes(
                    self.normalized[self.selected_normalized][6],
                    [shape for shape in self.normalized if shape[4] != self.normalized[self.selected_normalized][4]],
                    self.neighbor_count)
            if imgui.button("Get Best-Matching Shapes (Step 5: ANN)"):
                neighbor_indexes, distances = self.index.query(
                    np.array([self.normalized[self.selected_normalized][6].get_normalized_features()]),
                    k=self.neighbor_count
                )

                self.distances = distances.flatten().tolist()
                self.best_matching_shapes = [self.normalized[k] for k in neighbor_indexes.flatten().tolist()]

        # Step 6
        imgui.unindent(16)
        imgui.spacing()
        imgui.separator()
        imgui.text("Evaluation: ")
        imgui.spacing()
        imgui.indent(16)
        if imgui.button("Evaluate CBSR System"):

            self.shapes_per_class = calculate_shapes_per_class(self.normalized)
            self.show_rest_of_ui_step_5 = False
            self.show_rest_of_ui_step_6 = True

        if self.show_rest_of_ui_step_6:
            _, self.neighbor_count = imgui.input_int("Number of Returned Shapes", self.neighbor_count)

            if imgui.button("Use Custom Query"):
                self.precisions, self.recalls, self.f1_scores = evaluate_query(
                    "Custom", self.normalized, self.neighbor_count, self.shapes_per_class, None
                )
                self.evaluate = True

            if imgui.button("Use ANN"):
                self.precisions, self.recalls, self.f1_scores = evaluate_query(
                    "ANN", self.normalized, self.neighbor_count, self.shapes_per_class, self.index
                )
                self.evaluate = True

            if self.evaluate:
                imgui.set_next_window_position(0, 600, imgui.ONCE)

                if imgui.begin("Evaluation Results:", True):
                    if self.selected_evaluation_subject == 0:
                        evaluation_subject = "Average"
                    else:
                        evaluation_subject = list(self.shapes_per_class.keys())[self.selected_evaluation_subject - 1]
                    imgui.text(f"{evaluation_subject} Query Precision: {self.precisions[evaluation_subject]:.3f}")
                    imgui.text(f"{evaluation_subject} Query Recall: {self.recalls[evaluation_subject]:.3f}")
                    imgui.text(f"{evaluation_subject} Query F1 Score: {self.f1_scores[evaluation_subject]:.3f}")

                    imgui.spacing()
                    imgui.text("Select Evaluation Subject:")
                    imgui.spacing()
                    imgui.indent(16)
                    changed, self.selected_evaluation_subject = imgui.combo("",
                                                                            self.selected_evaluation_subject,
                                                                            ["Average"] + [shape_name for shape_name in
                                                                                           self.shapes_per_class.keys()])
                imgui.end()

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

            if not self.show_poorly_sampled and not self.show_normalized:
                self.average_model.color = [0, 0, 0]
                self.average_model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )

            elif self.show_poorly_sampled:
                model = self.poorly_sampled[self.selected_poorly_sampled][0]
                model.color = [0, 0, 0]
                model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )

                sample = self.refined[self.selected_poorly_sampled]
                model = sample[0]
                model.color = [0, 0, 0]
                model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )
                bb = sample[1]

                x_coordinates = [point[0][0] for point in bb]

                # Find the minimum and maximum x-coordinates
                min_x = min(x_coordinates)
                max_x = max(x_coordinates)

                # Calculate the width
                width = max_x - min_x
                model.translate(width + 1, 0)

            else:
                model = self.normalized[self.selected_normalized][0]
                model.color = [0, 0, 0]
                model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )
                model.translate(0, 0)

                translation = 2
                for match in self.best_matching_shapes:
                    model = match[0]
                    model.color = [0, 0, 0]
                    model.draw(
                        self.app.camera.projection.matrix,
                        self.app.camera.matrix,
                        self.light
                    )

                    model.translate(translation, 0)
                    translation *= -1
                    if translation > 0:
                        translation += 2

        self.app.ctx.wireframe = False

        if not self.show_poorly_sampled and not self.show_normalized:
            self.average_model.color = [1, 1, 1]
            self.average_model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )
        elif self.show_poorly_sampled:
            sample = self.poorly_sampled[self.selected_poorly_sampled]
            model = sample[0]
            model.color = [1, 1, 1]
            model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )

            self.model_bb = sample[1]
            self.current_model = model
            self.model_basis = sample[3] if self.move_axes_to_barycenter else sample[2]

            sample = self.refined[self.selected_poorly_sampled]
            model = sample[0]
            model.color = [1, 1, 1]
            model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )
            bb = sample[1]

            x_coordinates = [point[0][0] for point in bb]

            # Find the minimum and maximum x-coordinates
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)

            # Calculate the width
            width = max_x - min_x
            model.translate(width + 1, 0)

        else:
            sample = self.normalized[self.selected_normalized]
            model = sample[0]
            model.color = [1, 1, 1]
            model.draw(
                self.app.camera.projection.matrix,
                self.app.camera.matrix,
                self.light
            )
            self.model_bb = sample[1]
            self.current_model = model
            self.model_basis = sample[3] if self.move_axes_to_barycenter else sample[2]
            model.translate(0, 0)

            translation = 2
            for match in self.best_matching_shapes:
                model = match[0]
                model.color = [1, 1, 1]
                model.draw(
                    self.app.camera.projection.matrix,
                    self.app.camera.matrix,
                    self.light
                )

                model.translate(translation, 0)
                translation *= -1
                if translation > 0:
                    translation += 2

                model_bb = match[1]
                model_basis = match[3] if self.move_axes_to_barycenter else match[2]

                if self.show_bb:
                    self.lines.update(model_bb)
                    self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix,
                                    model.get_model_matrix())
                    self.lines.color = [0, 0, 1, 1]
                if self.show_axis:
                    self.lines.update(model_basis)
                    self.app.ctx.depth_func = '1'  # ALWAYS
                    self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix,
                                    model.get_model_matrix())
                    self.app.ctx.depth_func = '<'  # LESS
                    self.lines.color = [1, 0, 0, 1]

        if self.show_bb:
            self.lines.update(self.model_bb)
            self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix,
                            self.current_model.get_model_matrix())
            self.lines.color = [0, 0, 1, 1]
        if self.show_axis:
            self.lines.update(self.model_basis)
            self.app.ctx.depth_func = '1'  # ALWAYS
            self.lines.draw(self.app.camera.projection.matrix, self.app.camera.matrix,
                            self.current_model.get_model_matrix())
            self.app.ctx.depth_func = '<'  # LESS
            self.lines.color = [1, 0, 0, 1]

        self.render_ui()
