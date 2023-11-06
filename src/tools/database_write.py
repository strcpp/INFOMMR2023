from tqdm import tqdm
import trimesh
from multiprocessing import Pool, cpu_count
from descriptor_extraction import *
from display_statistics import return_neighbors

import pandas as pd

database_path = os.path.join('src', 'tools', 'outputs', 'database.csv')
models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Default')
normalized_models_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'models', 'Normalized')

THRESHOLD = 500


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


def align_to_largest_extent_component(mesh):
    extents = mesh.extents
    max_extent_direction = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    major_axis_index = np.argmax(extents)
    major_axis_sign = np.sign(max_extent_direction[major_axis_index])
    
    # Ensure the major axis points in the positive direction
    if major_axis_sign < 0:
        mesh.vertices[:, major_axis_index] *= -1  # This flips the mesh along the major axis

    return mesh

def align_mesh_axes(eig_vectors):
    # Reorder eigenvectors: longest axis (major) aligns with X-axis, shortest (minor) with Y-axis.
    # The cross product ensures that the resulting system is right-handed.
    major_axis_vector = eig_vectors[:, 2]  # longest axis - X
    minor_axis_vector = eig_vectors[:, 0]  # shortest axis - Y
    medium_axis_vector = np.cross(minor_axis_vector, major_axis_vector)  # cross product - Z
    
    # Now we construct the new orientation matrix such that each column is an axis vector
    new_orientation = np.column_stack((major_axis_vector, minor_axis_vector, medium_axis_vector))
    
    # Ensure the matrix is a proper rotation matrix by checking its determinant
    if np.linalg.det(new_orientation) < 0:
        new_orientation[:, 1] = -new_orientation[:, 1]  # invert Y if we don't have a right-handed system
    
    return new_orientation

def align_mesh_axes(eig_vectors, eig_values, vertices):
    # Sort eigenvalues and eigenvectors by the magnitude of the eigenvalues (major to minor)
    sort_idx = np.argsort(-eig_values)
    eig_vectors = eig_vectors[:, sort_idx]

    # Project the vertices onto the eigenvectors to find their extents
    projected_vertices = vertices @ eig_vectors
    min_extents = projected_vertices.min(axis=0)
    max_extents = projected_vertices.max(axis=0)
    extents = max_extents - min_extents

    # The axis with the largest extent should be aligned with the X-axis
    # The axis with the smallest extent should be aligned with the Y-axis
    axis_order = np.argsort(-extents)  # Sort axes by extent in descending order

    # Create a new set of eigenvectors with the correct order
    aligned_eig_vectors = eig_vectors[:, axis_order]

    # Ensure right-handed coordinate system
    cross_product = np.cross(aligned_eig_vectors[:, 0], aligned_eig_vectors[:, 1])
    if np.dot(cross_product, aligned_eig_vectors[:, 2]) < 0:
        aligned_eig_vectors[:, 2] = -aligned_eig_vectors[:, 2]

    return aligned_eig_vectors

def align_mesh(mesh, eig_vectors):
    # Apply the rotation to align the mesh with the new axes
    mesh.vertices = mesh.vertices @ eig_vectors

    # Ensure the axes are pointing in the positive direction based on the vertices' positions
    for i in range(3):
        if mesh.vertices[:, i].mean() < 0:
            mesh.vertices[:, i] = -mesh.vertices[:, i]

    return mesh


def process_mesh(args):
    m, target_faces = args
    model_class = m[1]
    model_name = m[2]
    mesh = m[0]

    # Step 1: Resample
    # Assuming 'resample' is a function defined elsewhere that you've imported
    mesh.process()
    mesh.remove_duplicate_faces()
    mesh = resample(mesh, target_faces)

   # Step 2: Translation
    barycenter = mesh.centroid
    mesh.apply_translation(-barycenter)


    # Step 3: Pose (alignment)
    covariance_matrix = np.cov(mesh.vertices.T)
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(-eig_values)
    eig_vectors = eig_vectors[:, sorted_indices]
    aligned_eig_vectors = align_mesh_axes(eig_vectors, eig_values, mesh.vertices)
    mesh = align_mesh(mesh, aligned_eig_vectors)

    # Step 4: Scale
    max_dimension = max(mesh.extents)
    scale_factor = 1.0 / max_dimension
    mesh.apply_scale(scale_factor)

    # Step 5: Export and Descriptors
    descriptors = ShapeDescriptors.from_mesh(mesh, model_class, model_name)  # assuming this is a predefined class
    base_model_name, _ = os.path.splitext(model_name)

    normalized_output_path = os.path.join(normalized_models_path, model_class)
    os.makedirs(normalized_output_path, exist_ok=True)

    # Construct the file path with the .obj extension explicitly added
    mesh_path = os.path.join(normalized_output_path, f"{base_model_name}.obj")
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
    with Pool(processes=cpu_count()) as pool:
        meshes = pool.map(load_model, paths_to_load)

    average_model, _ = return_neighbors()
    average_mesh = next((m[0] for m in meshes if m[2] == average_model["Shape Name"]), None)

    target_faces = len(average_mesh.vertices)

    # Use multiprocessing to parallelize the processing
    with Pool(processes=cpu_count()) as pool:
        all_descriptors = list(
            tqdm(pool.imap_unordered(process_mesh, [(m, target_faces) for m in meshes]), total=len(meshes)))
    data_list = []

    for descriptor in tqdm(all_descriptors, desc="Saving Descriptors for all Shapes", leave=False):
        data = {
            "Model Class": descriptor.model_class,
            "Model Name": descriptor.model_name,
            "Surface Area": round(descriptor.surface_area, 3),
            "Compactness": round(descriptor.compactness, 3),
            "Rectangularity": round(descriptor.rectangularity, 3),
            "Diameter": round(descriptor.diameter, 3),
            "Convexity": round(descriptor.convexity, 3),
            "Eccentricity": round(descriptor.eccentricity, 3),
            "A3": [round(x, 3) for x in descriptor.A3],
            "D1": [round(x, 3) for x in descriptor.D1],
            "D2": [round(x, 3) for x in descriptor.D2],
            "D3": [round(x, 3) for x in descriptor.D3],
            "D4": [round(x, 3) for x in descriptor.D4],
        }
        data_list.append(data)

    # headers = ["Model Class",
    #         "Model Name",
    #         "Surface Area",
    #         "Compactness",
    #         "Rectangularity",
    #         "Diameter",
    #         "Convexity",
    #         "Eccentricity",
    #         "A3",
    #         "D1",
    #         "D2",
    #         "D3",
    #         "D4"]
    #
    # try:
    #     with open(database_path, mode='w', newline='') as file:
    #         writer = csv.writer(file, delimiter=';')
    #         writer.writerow(headers)
    #
    #     # Append shape data to the CSV file
    #     with open(database_path, mode='a', newline='') as file:
    #         writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
    #         for shape in data_list:
    #             writer.writerow(shape)
    # except FileNotFoundError:
    #     try:
    #         path = os.path.join('tools', 'outputs', 'database.csv')
    #         with open(path, mode='w', newline='') as file:
    #             writer = csv.writer(file, delimiter=';')
    #             writer.writerow(headers)
    #
    #         # Append shape data to the CSV file
    #         with open(path, mode='a', newline='') as file:
    #             writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
    #             for shape in data_list:
    #                 writer.writerow(shape)
    #     except FileNotFoundError:
    #         path = os.path.join('outputs', 'database.csv')
    #         with open(path, mode='w', newline='') as file:
    #             writer = csv.writer(file, delimiter=';')
    #             writer.writerow(headers)
    #
    #         # Append shape data to the CSV file
    #         with open(path, mode='a', newline='') as file:
    #             writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
    #             for shape in data_list:
    #                 writer.writerow(shape)

    df = pd.DataFrame(data_list)

    try:
        df.to_csv(database_path, index=False, sep=';')
    except OSError:
        try:
            path = os.path.join('tools', 'outputs', 'database.csv')
            df.to_csv(path, index=False, sep=';')
        except OSError:
            path = os.path.join('outputs', 'database.csv')
            df.to_csv(path, index=False, sep=';')


if __name__ == '__main__':
    main()
