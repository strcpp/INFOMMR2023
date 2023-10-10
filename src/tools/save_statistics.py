import os
from tqdm import tqdm
import csv
import trimesh
import json

def save_refined(meshes):
    base = os.path.dirname(__file__)
    models_path =os.path.join(base, '../../resources/models')
    
    shape_data = []

    for key, mesh in meshes.items():
        bounding_box = mesh[2].bounds
        current_model_vertices = len(mesh[2].vertices) 
        current_model_faces = len(mesh[2].faces)
        
        # Save shape data
        if len(shape_data) == 0:
            shape_data = [
                {'Shape Name': mesh[0],
                    'Shape Class': mesh[1],
                    'Number of Vertices': current_model_vertices,
                    'Number of Faces': current_model_faces,
                    'Type of Faces': 'Triangle',
                    '3D Bounding Box': bounding_box}
            ]
        else:
            shape_data.append({'Shape Name': mesh[0],
                                'Shape Class': mesh[1],
                                'Number of Vertices': current_model_vertices,
                                'Number of Faces': current_model_faces,
                                'Type of Faces': 'Triangle',
                                '3D Bounding Box': bounding_box})
    
    # Path to the CSV file
    csv_file_path = os.path.join('outputs', 'shape_data_resampled.csv')

    # CSV file headers
    headers = ['Shape Name', 'Shape Class', 'Number of Vertices', 'Number of Faces', 'Type of Faces', '3D Bounding Box']

    csv_path = os.path.join(base, csv_file_path)
    # Create CSV file and add headers
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(headers)

    # Append shape data to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
        for shape in shape_data:
            writer.writerow(shape)       

def save():
    """
    Saves shape data (filename, class name, number of vertices, number of faces, type of faces) to a CSV file
    """
    base = os.path.dirname(__file__)
    models_path =os.path.join(base, '../../resources/models')
    
    shape_data = []

    # How many files to load from each class, usually to speed up devel
    try:
        with open('config.json', 'r') as file:
                config = json.load(file)
    except FileNotFoundError:
        with open('../../config.json', 'r') as file:
                config = json.load(file)

    # Iterate through all .obj files
    for root, dirs, files in tqdm(os.walk(models_path), desc="Parsing .obj files"):
        if(len(files) > 0):
            len_files = len(files) if config['len_files'] == 0 else config['len_files']

            for i in range(len_files):
                file = files[i]
                current_class = os.path.basename(os.path.normpath(root))
                file_path = os.path.join(root, file)

                # Get axis-aligned 3D bounding box
                mesh = trimesh.load_mesh(file_path)
                bounding_box = mesh.bounds
                number_of_vertices = len(mesh.vertices)
                number_of_faces = len(mesh.faces)

                if len(shape_data) == 0:
                    shape_data = [
                        {'Shape Name': file,
                         'Shape Class': current_class,
                         'Number of Vertices': number_of_vertices,
                         'Number of Faces': number_of_faces,
                         'Type of Faces': 'Triangle',
                         '3D Bounding Box': bounding_box}
                    ]
                else:
                    shape_data.append({'Shape Name': file,
                                       'Shape Class': current_class,
                                       'Number of Vertices': number_of_vertices,
                                       'Number of Faces': number_of_faces,
                                       'Type of Faces': 'Triangle',
                                       '3D Bounding Box': bounding_box})
    # Path to the CSV file
    csv_file_path = os.path.join('outputs', 'shape_data.csv')

    # CSV file headers
    headers = ['Shape Name', 'Shape Class', 'Number of Vertices', 'Number of Faces', 'Type of Faces', '3D Bounding Box']

    csv_path = os.path.join(base, csv_file_path)
    # Create CSV file and add headers
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(headers)

    # Append shape data to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
        for shape in shape_data:
            writer.writerow(shape)


if __name__ == '__main__':
    save()
