from __future__ import annotations
import trimesh
import numpy as np
from numba import njit
from tools.descriptor_extraction import ShapeDescriptors
from pynndescent import NNDescent
from tqdm import tqdm

THRESHOLD = 300


def get_bb_lines(bounding_box: np.ndarray[float]) -> list[tuple[float, float]]:
    """ Gets the bounding box connections of a shape's bounding box coordinates. """
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


def get_basis_lines(bounding_box: np.ndarray, barycenter: np.ndarray) -> list[tuple[any, any]]:
    """ Returns the 3D axis of a shape calculated at its center. """
    # Calculate the model's geometric center
    if bounding_box is not None:
        center_x = (bounding_box[0][0] + bounding_box[1][0]) / 2
        center_y = (bounding_box[0][1] + bounding_box[1][1]) / 2
        center_z = (bounding_box[0][2] + bounding_box[1][2]) / 2
    else:
        center_x, center_y, center_z = barycenter

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


def resample(mesh: trimesh.Trimesh, target_vertices: int) -> trimesh.Trimesh:
    """ Resamples a mesh to a specific target number of vertices. """
    while len(mesh.vertices) > target_vertices + THRESHOLD or len(mesh.vertices) < target_vertices - THRESHOLD:
        # If number of vertices is too high, simplify
        if len(mesh.vertices) > target_vertices + THRESHOLD:
            new_face_count = target_vertices * (len(mesh.faces) / len(mesh.vertices))
            mesh = mesh.simplify_quadratic_decimation(new_face_count)

        # If number of vertices is too low, subdivide
        if len(mesh.vertices) < target_vertices - THRESHOLD:
            mesh = trimesh.Trimesh(*trimesh.remesh.subdivide(mesh.vertices, mesh.faces))
    return mesh


def normalize_single_features(mesh_features: ShapeDescriptors) -> None:
    """ Normalizes the single features of all shapes. """
    feature_values = [descriptor.get_single_features() for descriptor in mesh_features]

    features_array = np.array(feature_values)

    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0)

    standardized_features = (features_array - mean) / std

    for i, descriptor in enumerate(mesh_features):
        descriptor.normalize_single_features(standardized_features[i])


@njit
def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """ Calculates the Euclidean distance of 2 features. """
    return np.sqrt(np.sum((x1 - x2) ** 2))

from scipy.stats import wasserstein_distance




def get_best_matching_shapes(current_mesh, all_meshes, num_neighbors):
    distances = {}
    current_features = np.array(current_mesh.get_normalized_features())
    #current_features2 = np.array(current_mesh.get_normalized_features2())

    for model_name, mesh in all_meshes.items():
        mesh_features = np.array(mesh.get_normalized_features())
        distances[model_name] = euclidean_distance(current_features, mesh_features)


       # mesh_features = np.array(mesh.get_normalized_features2())
        #distances[model_name] += euclidean_distance(current_features2, mesh_features)

    sorted_distances = sorted(distances.items(), key=lambda item: item[1])
    best_matching_shapes = [model_name for model_name, _ in sorted_distances[:num_neighbors]]

    return best_matching_shapes, sorted_distances


def calculate_shapes_per_class(shapes: list[any]) -> dict[str, int]:
    """ Calculates the number of shapes of each shape class. """
    shapes_per_class = {}
    for shape in shapes:
        shape_class = shape[5]
        if shape_class in shapes_per_class:
            shapes_per_class[shape_class] += 1
        else:
            shapes_per_class[shape_class] = 1
    return shapes_per_class


def calculate_precision(name: str, matched_shapes: list[any], all_classes: dict[str, list[str]]) -> tuple[float, str]:
    """ Calculates the precision of a query. """
    tp = 0
    fp = 0
    for key, values in all_classes.items():
        if name in values:
            correct_class = key
            break
    for matched_shape in matched_shapes:
        if matched_shape in all_classes.get(correct_class, []):
            tp += 1
        else:
            fp += 1
    return tp / (tp + fp) if (tp + fp) > 0 else 0, correct_class


def calculate_recall(name: str, matched_shapes: list[any], all_classes: dict[str, list[str]]) -> tuple[float, str]:
    """ Calculates the recall of a query. """
    tp = 0
    for key, values in all_classes.items():
        if name in values:
            correct_class = key
            break
    for matched_shape in matched_shapes:
        if matched_shape in all_classes.get(correct_class, []):
            tp += 1
    fn = len(all_classes.get(correct_class, [])) - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0, correct_class


def evaluate_query(
        query_type: str, all_shapes, k: int, shapes_per_class: dict[str, int], all_classes: dict[str, list[str]],
        index: NNDescent
):
    """ Evaluates quality of selected query. """
    precisions = {}
    recalls = {}
    f1_scores = {}
    average_precision = 0
    average_recall = 0
    # Query all shapes
    for name, descriptor in tqdm(all_shapes.items(),
                                 desc=f"Finding the {k} Best Matching Shapes for each Shape", leave=False):
        # Query based on selected query type
        if query_type == "Custom":
            matching_names, _ = get_best_matching_shapes(
                descriptor, {key: value for key, value in all_shapes.items() if key != name}, k
            )
        elif query_type == "ANN":
            neighbor_indexes, _ = index.query(np.array([descriptor.get_normalized_features()]), k=k + 1)
            matching_names = [list(all_shapes.keys())[k] for k in neighbor_indexes.flatten().tolist()[1:]]

        precision, correct_class = calculate_precision(name, matching_names, all_classes)
        recall, _ = calculate_recall(name, matching_names, all_classes)
        average_precision += precision
        average_recall += recall

        if correct_class in precisions:
            precisions[correct_class] += precision
            recalls[correct_class] += recall
        else:
            precisions[correct_class] = precision
            recalls[correct_class] = recall

    # Calculate averages
    for shape_class in shapes_per_class.keys():
        precisions[shape_class] /= shapes_per_class[shape_class]
        recalls[shape_class] /= shapes_per_class[shape_class]

    average_precision /= len(all_shapes)
    average_recall /= len(all_shapes)

    precision_sum_recall = average_recall + average_precision
    if precision_sum_recall > 0:
        f1_score = 2 * average_precision * average_precision / precision_sum_recall
    else:
        f1_score = 0

    precisions["Average"] = average_precision
    recalls["Average"] = average_recall
    f1_scores["Average"] = f1_score

    for shape_class in precisions.keys():
        precision_sum_recall = precisions[shape_class] + recalls[shape_class]
        if precision_sum_recall > 0:
            f1_scores[shape_class] = 2 * precisions[shape_class] * recalls[shape_class] / precision_sum_recall
        else:
            f1_scores[shape_class] = 0

    return precisions, recalls, f1_scores
