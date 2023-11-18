from __future__ import annotations
from render.model import Model
from render.skybox import Skybox
from scenes.scene import Scene
from pyrr import Vector3
from light import Light
import imgui
import os
from tools.display_statistics import return_neighbors, return_bounding_box, return_shape_descriptors
from tqdm import tqdm
from render.lines import Lines
import trimesh
import numpy as np
from tools.descriptor_extraction import *
from multiprocessing import Pool, cpu_count
from scenes.scene_utils import *
import pynndescent
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mplcursors


def normalize_single_features(mesh_features: dict[str, ShapeDescriptors]) -> None:
    """
    Normalizes the single features of a shape.
    :param mesh_features: Features of the shape.
    """
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


def load_model(path: tuple[str, str, str]) -> tuple[trimesh.Trimesh, str, str]:
    """
    Load model from path.
    :param path: Tuple containing the model's path, name and class.
    :return: Tuple containing the model's mesh, name and class.
    """
    mesh = trimesh.load_mesh(path[0])
    return mesh, path[1], path[2]


class QueryScene(Scene):
    """
    Implements the scene of the application.
    """
    # Paths
    original_model_path = os.path.join(os.path.dirname(__file__), '../../resources/models/Default')
    normalized_model_path = os.path.join(os.path.dirname(__file__), '../../resources/models/Normalized')
    # Use this instead of "normalized_model_path" to load the normalized models from the directory with fewer classes
    less_class_model_path = os.path.join(os.path.dirname(__file__), '../../resources/models/Normalized_Less_Classes')

    # Rendering
    light = None
    skybox = None
    lines = None
    current_shading_mode = "flat"

    # Models
    models = {}
    current_model = None
    current_class = ""
    current_model_name = ""
    current_model_id = 0
    current_class_id = 0
    average_model = None
    average_vertices = 0
    average_faces = 0
    average_model_class = ""
    model_bb = []
    model_basis = None
    all_classes = []
    poorly_sampled = []
    refined = []
    models_of_current_class = None

    all_model_names = {}
    all_meshes = {}
    all_descriptors = {}
    all_basis_lines = {}
    all_barycenter_lines = {}
    all_bounding_boxes = {}
    current_descriptor = None

    # UI
    show_wireframe = False
    move_axes_to_barycenter = False
    show_bb = False
    show_axis = False
    evaluate = False
    show_poorly_sampled = False
    show_normalized = False
    selected_normalized = False
    evaluate_cbsr = False
    selected_class = 0
    selected_model = 0
    selected_poorly_sampled = 0
    selected_distance_id = 0
    selected_distance = "Euclidean"
    available_distances = ["Euclidean",
                           "Cosine",
                           "EMD",
                           "Euclidean (Single) + EMD (Histogram)",
                           "Euclidean (Single) + Cosine (Histogram)",
                           "Cosine (Single) + EMD (Histogram)",
                           "Cosine (Single) + Euclidean (Histogram)",
                           "EMD (Single) + Euclidean (Histogram)",
                           "EMD (Single) + Cosine (Histogram)",
                           ]

    # Query
    index = None
    distances = None
    neighbor_count = 1
    best_matching_shapes = []
    shapes_per_class = {}
    selected_evaluation_subject = 0
    precisions, recalls, f1_scores = {}, {}, {}

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

        # Conditions for poorly sampled shapes
        condition_low = (all_neighbors['Number of Vertices'] < 100)
        condition_high = (all_neighbors['Number of Vertices'] > 50000)

        # Use 5 shapes with less than 100 faces/vertices and 5 shapes with more than 50000 faces/vertices
        poorly_sampled = pd.concat([all_neighbors[condition_low].head(4), all_neighbors[condition_high].head(4)])

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

        self.light = Light(
            position=Vector3([5., 5., 5.], dtype='f4'),
            color=Vector3([1.0, 1.0, 1.0], dtype='f4')
        )

        self.current_shading_mode = 0
        self.lines = Lines(self.app, line_width=1)

        normalize_single_features(self.all_descriptors)

        # Setting up ANN query
        print("< Setting up ANN Query...")
        query_shapes = np.array([descriptor.get_weighted_normalized_features() for descriptor in self.all_descriptors.values()])

        # Setting all NaN features to 0
        query_shapes = np.nan_to_num(query_shapes, nan=0)

        # Preparing index
        self.index = pynndescent.NNDescent(query_shapes, n_jobs=-1, random_state=42, metric="braycurtis")
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
        imgui.spacing()
        imgui.text("Visualization Settings: ")
        imgui.spacing()
        imgui.indent(16)

        _, self.show_wireframe = imgui.checkbox("Show Wireframe", self.show_wireframe)
        _, self.show_bb = imgui.checkbox("Show Bounding Boxes", self.show_bb)
        _, self.show_axis = imgui.checkbox("Show Axes", self.show_axis)

        # Render 3D axes
        if self.show_axis:
            imgui.spacing()
            imgui.indent(16)
            # Move 3D axes to the shape's barycenter
            _, self.move_axes_to_barycenter = imgui.checkbox("Move Axes to Barycenter",
                                                             self.move_axes_to_barycenter)
            imgui.unindent(16)

        # Step 2 - Show outliers and their resampled counterparts
        imgui.unindent(16)
        imgui.spacing()
        imgui.separator()
        imgui.text("Step 2 - Outlier Resampling: ")
        imgui.spacing()
        _, self.show_poorly_sampled = imgui.checkbox("Show Poorly Sampled Shapes",
                                                     self.show_poorly_sampled)

        if self.show_poorly_sampled:
            imgui.spacing()
            imgui.indent(16)
            imgui.text("Select Poorly Sampled Shape:")
            imgui.spacing()
            imgui.indent(16)
            _, self.selected_poorly_sampled = imgui.combo(" ",
                                                          self.selected_poorly_sampled,
                                                          [f"{shape[4]}({shape[5]})" for shape in self.poorly_sampled])
            imgui.unindent(32)
            self.show_normalized = False
            self.evaluate_cbsr = False

            self.current_model_name = self.poorly_sampled[self.selected_poorly_sampled][4]
            self.current_class = self.poorly_sampled[self.selected_poorly_sampled][5]
            self.current_model = self.poorly_sampled[self.selected_poorly_sampled][0]
            mesh = self.current_model.mesh

            self.current_descriptor = self.all_descriptors[self.current_model_name]
            self.all_bounding_boxes[self.current_model_name] = get_bb_lines(mesh.bounds)
            self.all_basis_lines[self.current_model_name] = get_basis_lines(mesh.bounds, None)
            self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None, mesh.centroid)

            self.selected_normalized = False

        # Step 3
        imgui.spacing()
        imgui.separator()
        imgui.text("Step 3 - Normalization: ")
        imgui.spacing()
        _, self.show_normalized = imgui.checkbox("Show Normalized Shapes", self.show_normalized)

        if self.show_normalized:
            imgui.indent(16)
            imgui.spacing()
            imgui.text("Select Normalized Shape:")
            imgui.spacing()
            imgui.indent(16)
            self.show_poorly_sampled = False
            self.evaluate_cbsr = False

            # By default, show 1st shape of 1st class
            if not self.selected_normalized:
                self.current_class_id = 0
                self.current_class = self.all_classes[self.current_class_id]
                self.current_model_name = self.all_model_names[self.current_class][0]
                mesh = self.all_meshes[self.current_model_name]
                self.current_model = Model(self.app, self.current_model_name, mesh)
                self.current_descriptor = self.all_descriptors[self.current_model_name]
                self.all_bounding_boxes[self.current_model_name] = get_bb_lines(
                    self.all_meshes[self.current_model_name].bounds)
                self.all_basis_lines[self.current_model_name] = get_basis_lines(
                    self.all_meshes[self.current_model_name].bounds, None)
                self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None, mesh.centroid)

            # Add a combo box for classes
            clicked, self.current_class_id = imgui.combo("Classes", self.current_class_id, self.all_classes)
            if clicked:
                self.current_class = self.all_classes[self.current_class_id]
                self.current_model_name = self.all_model_names[self.current_class][0]
                mesh = self.all_meshes[self.current_model_name]
                self.current_model = Model(self.app, self.current_model_name, mesh)
                self.current_descriptor = self.all_descriptors[self.current_model_name]
                self.all_bounding_boxes[self.current_model_name] = get_bb_lines(
                    self.all_meshes[self.current_model_name].bounds)
                self.all_basis_lines[self.current_model_name] = get_basis_lines(
                    self.all_meshes[self.current_model_name].bounds, None)
                self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None, mesh.centroid)

                self.models_of_current_class = sorted(self.all_model_names[self.current_class])
                self.current_model_id = 0
                self.selected_normalized = True

            # Add a combo box for models based on selected class
            if self.current_class and self.current_class in self.all_model_names:
                self.models_of_current_class = sorted(self.all_model_names[self.current_class])

                clicked, self.current_model_id = imgui.combo("Models", self.current_model_id,
                                                             self.models_of_current_class)
                if clicked:
                    self.current_model_name = self.models_of_current_class[self.current_model_id]
                    mesh = self.all_meshes[self.current_model_name]
                    self.current_model = Model(self.app, self.current_model_name, mesh)
                    self.current_descriptor = self.all_descriptors[self.current_model_name]
                    self.all_bounding_boxes[self.current_model_name] = get_bb_lines(
                        self.all_meshes[self.current_model_name].bounds)
                    self.all_basis_lines[self.current_model_name] = get_basis_lines(
                        self.all_meshes[self.current_model_name].bounds, None)
                    self.all_barycenter_lines[self.current_model_name] = get_basis_lines(None, mesh.centroid)

        if self.show_normalized:
            imgui.unindent(32)
            imgui.spacing()
            imgui.separator()
            imgui.text("Steps 4-5: Query: ")
            imgui.spacing()
            imgui.indent(16)

            imgui.text("Select the Number of Returned Shapes: ")
            imgui.spacing()
            imgui.indent(16)
            _, self.neighbor_count = imgui.input_int(" ", self.neighbor_count)
            imgui.unindent(16)
            imgui.spacing()
            imgui.text("Select Distance Metric (Custom): ")
            imgui.spacing()
            imgui.indent(16)
            clicked, self.selected_distance_id = imgui.combo("      ", self.selected_distance_id,
                                                             [distance for distance in self.available_distances])
            imgui.unindent(16)
            if clicked:
                self.selected_distance = self.available_distances[self.selected_distance_id]
            imgui.spacing()
            if imgui.button("Get Best-Matching Shapes (Custom)"):
                matching_names, self.distances = get_best_matching_shapes(
                    self.all_descriptors[self.current_model_name],
                    {key: value for key, value in self.all_descriptors.items() if key != self.current_model_name},
                    self.neighbor_count, self.selected_distance)
                self.best_matching_shapes = [(Model(self.app, name, self.all_meshes[name]), name) for name in
                                             matching_names]
                for name in matching_names:
                    self.all_bounding_boxes[name] = get_bb_lines(self.all_meshes[name].bounds)
                    self.all_basis_lines[name] = get_basis_lines(self.all_meshes[name].bounds, None)
                    self.all_barycenter_lines[name] = get_basis_lines(None, self.all_meshes[name].centroid)

            if imgui.button("Get Best-Matching Shapes (ANN)"):
                neighbor_indexes, distances = self.index.query(
                    np.array([self.all_descriptors[self.current_model_name].get_weighted_normalized_features()]),
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
            imgui.unindent(16)

        if self.current_model != '':
            imgui.set_next_window_position(420, 20, imgui.ONCE)
            if imgui.begin("Info", True):
                # Show information about the selected poorly-sample shape (left) and its resampled counterpart (right)
                if self.show_poorly_sampled:
                    imgui.columns(2, None, True)
                    # Resampled shape info
                    imgui.set_column_width(-1, 250)
                    sample = self.refined[self.selected_poorly_sampled]
                    imgui.text(f"Shape Class: {sample[5]}")
                    mesh = sample[0].mesh
                    imgui.text(f"Number of Vertices: {len(mesh.vertices)}")
                    imgui.text(f"Number of Faces: {len(mesh.faces)}")
                    imgui.next_column()
                    # Poorly-sampled shape info
                    imgui.set_column_width(-1, 250)
                    sample = self.poorly_sampled[self.selected_poorly_sampled]
                    imgui.text(f"Shape Class: {sample[5]}")
                    imgui.text(f"Number of Vertices: {len(sample[0].mesh.vertices)}")
                    imgui.text(f"Number of Faces: {len(sample[0].mesh.faces)}")
                    imgui.text("")

                # Show information about the normalized shape (center) and its n best matching shapes
                elif self.show_normalized:
                    imgui.columns(4, None, True)
                    aligned_shapes = [self.all_meshes[self.current_model_name].metadata['file_name']]
                    aligned_distances = [0]
                    for i in range(0, len(self.best_matching_shapes), 2):
                        try:
                            shape2 = self.best_matching_shapes[i + 1][1]
                            aligned_shapes.insert(len(aligned_shapes), shape2)
                            aligned_distances.insert(len(aligned_shapes), self.distances[i + 1])
                        except IndexError:
                            pass
                        shape1 = self.best_matching_shapes[i][1]
                        aligned_shapes.insert(0, shape1)
                        aligned_distances.insert(0, self.distances[i])

                    for i, shape in enumerate(aligned_shapes):
                        imgui.set_column_width(-1, 280)
                        current_class = ""
                        for shape_class, shape_list in self.all_model_names.items():
                            if shape in shape_list:
                                current_class = shape_class
                                break
                        imgui.text(f"Shape Class: {current_class}")
                        imgui.text(f"Number of Vertices: {len(self.all_meshes[shape].vertices)}")
                        imgui.text(f"Number of Faces: {len(self.all_meshes[shape].faces)}")
                        if len(aligned_distances) > 1:
                            imgui.text(f"Distance to Query Shape: {round(aligned_distances[i], 3)}")
                        imgui.text("{}: {:.2f}".format("Surface area", self.all_descriptors[shape].surface_area))
                        imgui.text("{}: {:.2f}".format("Compactness", self.all_descriptors[shape].compactness))
                        imgui.text("{}: {:.2f}".format("Rectangularity",
                                                       self.all_descriptors[shape].rectangularity))
                        imgui.text("{}: {:.2f}".format("Diameter", self.all_descriptors[shape].diameter))
                        imgui.text("{}: {:.2f}".format("Convexity", self.all_descriptors[shape].convexity))
                        imgui.text("{}: {:.2f}".format("Eccentricity", self.all_descriptors[shape].eccentricity))
                        imgui.spacing()
                        imgui.spacing()
                        imgui.spacing()
                        imgui.spacing()
                        imgui.spacing()
                        imgui.next_column()
                else:
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

            # Step 5
            imgui.spacing()
            imgui.separator()
            imgui.text("Step 5: Scalability")
            imgui.spacing()
            imgui.indent(16)
            if imgui.button("Run t-SNE"):
                model_names = [name for name in self.all_descriptors]

                reduced_features = TSNE(
                    n_components=2, learning_rate='auto', init='random', perplexity=15, n_jobs=-1, random_state=42
                ).fit_transform(np.array([mesh.get_weighted_normalized_features() for _, mesh in self.all_descriptors.items()]))

                # Assign colors to classes
                unique_classes = list(set(self.all_classes))
                num_classes = len(unique_classes)
                class_colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

                colormap = []
                model_classes = []

                for name in model_names:
                    for index, (shape_class, shape_names) in enumerate(self.all_model_names.items()):
                        for shape_name in shape_names:
                            if shape_name == name:
                                colormap.append(class_colors[index])
                                model_classes.append(shape_class)
                                break
                        else:
                            continue
                        break

                # Create scatterplot
                plt.subplots(figsize=(16, 9))
                plt.scatter(reduced_features[:, 0], reduced_features[:, 1], s=5, color=colormap)
                plt.title('t-SNE Scatterplot')
                plt.xlabel('First Feature')
                plt.ylabel('Second Feature')

                # Hover on points to see their shape class and name
                cursor = mplcursors.cursor(hover=True)
                cursor.connect("add", lambda sel: sel.annotation.set_text(
                    f"Class: {model_classes[sel.target.index]}\nName: {model_names[sel.target.index]}"))

                plt.show()

            # Step 6
            imgui.unindent(16)
            imgui.spacing()
            imgui.separator()
            imgui.text("Step 6: Evaluation")
            imgui.spacing()

            _, self.evaluate_cbsr = imgui.checkbox("Evaluate CBSR System", self.evaluate_cbsr)

            if self.evaluate_cbsr:
                imgui.indent(16)
                self.shapes_per_class = {key: len(value) for key, value in self.all_model_names.items()}
                self.show_poorly_sampled = False
                self.show_normalized = False

                imgui.spacing()
                imgui.text("Select the Number of Returned Shapes: ")
                imgui.spacing()
                imgui.indent(16)
                _, self.neighbor_count = imgui.input_int(" ", self.neighbor_count)
                imgui.unindent(16)
                imgui.spacing()
                imgui.text("Select Distance Metric (Custom): ")
                imgui.spacing()
                imgui.indent(16)
                clicked, self.selected_distance_id = imgui.combo("              ", self.selected_distance_id,
                                                                 [distance for distance in self.available_distances])
                imgui.spacing()
                imgui.unindent(16)
                if clicked:
                    self.selected_distance = self.available_distances[self.selected_distance_id]

                if imgui.button("Evaluate Custom Query"):
                    self.precisions, self.recalls, self.f1_scores = evaluate_query(
                        "Custom", self.all_descriptors, self.neighbor_count, self.shapes_per_class,
                        self.all_model_names, None, self.selected_distance
                    )
                    self.evaluate = True

                if imgui.button("Evaluate ANN"):
                    self.precisions, self.recalls, self.f1_scores = evaluate_query(
                        "ANN", self.all_descriptors, self.neighbor_count, self.shapes_per_class, self.all_model_names,
                        self.index, ""
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

        if not self.show_normalized and not self.show_poorly_sampled:
            self.current_model_name = self.average_model["Shape Name"]
            self.current_model = Model(self.app, self.current_model_name, self.all_meshes[self.current_model_name])
            self.current_class = self.average_model["Shape Class"]

        imgui.end()
        imgui.render()

        self.app.imgui.render(imgui.get_draw_data())

    def render_shapes(self, color: list[int] = [1, 1, 1, 1]) -> None:
        """
        Renders one or more shapes.
        :param color: Shape color.
        """

        def draw_shape(model: Model, name: str, translation: float = 0) -> None:
            """
            Renders a shape.
            :param model: Model that will be rendered.
            :param name: Model name.
            :param translation: Tranlsation of the model that will be rendered.
            """
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

            if self.show_normalized:
                # Render all the best-matching shapes that resulted from a query
                translation = 2
                for match in self.best_matching_shapes:
                    draw_shape(match[0], match[1], translation)
                    translation *= -1
                    if translation > 0:
                        translation += 2
            elif self.show_poorly_sampled:
                # Render refined shape next to its poorly-sampled counterpart
                sample = self.refined[self.selected_poorly_sampled]
                bb = sample[1]
                x_coordinates = [point[0][0] for point in bb]
                min_x = min(x_coordinates)
                max_x = max(x_coordinates)
                width = max_x - min_x
                translation = width + 1
                draw_shape(sample[0], sample[4], translation)

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
