import csv
from tqdm import tqdm
import trimesh
from multiprocessing import Pool, cpu_count
from descriptor_extraction import *
from display_statistics import return_neighbors, return_bounding_box

import pandas as pd

database_path = os.path.join('src', 'tools', 'outputs', 'database.csv')
models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Default')
normalized_models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Normalized')

THRESHOLD = 100


def resample(mesh, target_faces):
    while len(mesh.vertices) > target_faces + THRESHOLD or len(mesh.vertices) < target_faces - THRESHOLD:
        # If number of vertices is too high, simplify
        if len(mesh.vertices) > target_faces + THRESHOLD:
            new_face_count = target_faces * (len(mesh.faces) / len(mesh.vertices))
            mesh = mesh.simplify_quadratic_decimation(new_face_count)

        # If number of vertices is too low, subdivide
        if len(mesh.vertices) < target_faces - THRESHOLD:
            mesh = trimesh.Trimesh(*trimesh.remesh.subdivide(mesh.vertices, mesh.faces))
    return mesh


def process_mesh(args):
    m, target_faces = args  # unpack the tuple here
    model_class = m[1]
    model_name = m[2]
    mesh = m[0]

    # Step 1: Resample
    mesh.process()
    mesh.remove_duplicate_faces()
    mesh = resample(mesh, target_faces)

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
    descriptors = ShapeDescriptors.from_mesh(mesh, model_class, model_name)

    normalized_output_path = os.path.join(normalized_models_path, model_class)
    if not os.path.exists(normalized_output_path):
        os.makedirs(normalized_output_path)
    mesh_path = os.path.join(normalized_output_path, model_name)
    mesh.export(mesh_path, file_type="obj")

    return descriptors


def load_model(path):
    mesh = trimesh.load_mesh(path[0])
    return mesh, path[1], path[2]


def main():
    # Store all paths to be processed
    paths_to_load = []

    for root, dirs, files in os.walk(models_path):
        len_files = len(files)
        if len(files) > 0:
            for i in range(len_files):
                file = files[i]
                model_class = os.path.basename(os.path.normpath(root))
                path = os.path.join(models_path, model_class, file)
                paths_to_load.append((path, model_class, file))

    # Use multiprocessing to parallelize the loading
    with Pool(processes=cpu_count() - 4) as pool:
        meshes = pool.map(load_model, paths_to_load)

    average_model, _ = return_neighbors()
    average_mesh = next((m[0] for m in meshes if m[2] == average_model["Shape Name"]), None)
    target_faces = len(average_mesh.vertices)

    # Use multiprocessing to parallelize the processing
    with Pool(processes=cpu_count()) as pool:
        all_descriptors = list(
            tqdm(pool.imap_unordered(process_mesh, [(m, target_faces) for m in meshes]), total=len(meshes)))
    data_list = []

    for descriptor in tqdm(all_descriptors, desc="Calculating Descriptors for all Shapes", leave=False):
        data = {
            "Model Class": descriptor.model_class,
            "Model Name": descriptor.model_name,
            "Surface Area": descriptor.surface_area,
            "Compactness": descriptor.compactness,
            "Rectangularity": descriptor.rectangularity,
            "Diameter": descriptor.diameter,
            "Convexity": descriptor.convexity,
            "Eccentricity": descriptor.eccentricity,
            "A3": descriptor.A3,
            "D1": descriptor.D1,
            "D2": descriptor.D2,
            "D3": descriptor.D3,
            "D4": descriptor.D4,
        }
        data_list.append(data)

    df = pd.DataFrame(data_list)
    try:
        df.to_csv(database_path, index=False)
    except OSError:
        try:
            path = os.path.join('tools', 'outputs', 'database.csv')
            df.to_csv(path, index=False)
        except OSError:
            path = os.path.join('outputs', 'database.csv')
            df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
