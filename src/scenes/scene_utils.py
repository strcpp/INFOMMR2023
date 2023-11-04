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
    target_vertices = 1000
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


def get_best_matching_shapes(
        current_mesh: ShapeDescriptors, all_meshes: list[any], num_neighbors: int
) -> tuple[list[any], list[float]]:
    """ Finds the k best matching shapes for a query shape. """
    distances = np.zeros(len(all_meshes))
    for i in range(len(all_meshes)):
        x1 = np.array(current_mesh.get_normalized_features())
        x2 = np.array(all_meshes[i][6].get_normalized_features())
        distances[i] = euclidean_distance(x1, x2)
    sorted_distances = np.argsort(distances)[:num_neighbors]
    best_matching_shapes = [all_meshes[k] for k in sorted_distances]
    best_distances = [distances[k] for k in sorted_distances]
    return best_matching_shapes, best_distances


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


def calculate_precision(shape_class: str, matched_shapes: list[any]) -> float:
    """ Calculates the precision of a query. """
    tp = 0
    fp = 0
    for shape in matched_shapes:
        if shape[5] == shape_class:
            tp += 1
        else:
            fp += 1
    return tp / (tp + fp)


def calculate_recall(shape_class: str, shape_count: int, matched_shapes: list[any]) -> float:
    """ Calculates the recall of a query. """
    tp = 0
    for shape in matched_shapes:
        if shape[5] == shape_class:
            tp += 1
    fn = shape_count - tp
    return tp / (tp + fn)


def evaluate_query(query_type: str, all_shapes, k: int, shapes_per_class: dict[str, int], index: NNDescent):
    """ Evaluates quality of selected query. """
    precisions = {}
    recalls = {}
    f1_scores = {}
    average_precision = 0
    average_recall = 0
    # Query all shapes
    for query_shape in tqdm(all_shapes, desc=f"Finding the {k} Best Matching Shapes", leave=False):

        # Query based on selected query type
        if query_type == "Custom":
            best_matching_shapes, _ = get_best_matching_shapes(
                query_shape[6], [shape for shape in all_shapes if shape[4] != query_shape[4]], k
            )
        elif query_type == "ANN":
            neighbor_indexes, _ = index.query(np.array([query_shape[6].get_normalized_features()]), k=k)
            best_matching_shapes = [all_shapes[n] for n in neighbor_indexes.flatten().tolist()]

        precision = calculate_precision(query_shape[5], best_matching_shapes)
        recall = calculate_recall(query_shape[5],
                                  shapes_per_class[query_shape[5]],
                                  best_matching_shapes)
        average_precision += precision
        average_recall += recall
        if query_shape[5] in precisions:
            precisions[query_shape[5]] += precision
            recalls[query_shape[5]] += recall
        else:
            precisions[query_shape[5]] = precision
            recalls[query_shape[5]] = recall

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
