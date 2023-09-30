import os
from tqdm import tqdm
import csv
import trimesh


def main():
    """
    Saves shape data (filename, class name, number of vertices, number of faces, type of faces) to a CSV file
    """
    models_path = os.path.join(r"C:\Users\Xaris\Desktop\INFOMMR2023\resources\models")

    shape_data = []

    # Iterate through all .obj files
    for root, dirs, files in tqdm(os.walk(models_path), desc="Parsing .obj files"):
        for file in files:
            current_model_vertices = 0
            current_model_faces = 0
            current_class = os.path.basename(os.path.normpath(root))
            file_path = os.path.join(root, file)

            # Get axis-aligned 3D bounding box
            mesh = trimesh.load_mesh(file_path)
            bounding_box = mesh.bounds

            with open(file_path, 'r') as f:
                content = f.read()
                if '# Vertices:' in content:  # File is a m###.obj file
                    f.seek(72)  # Hardcoding this in order to speed-up file read
                    for line in f:
                        if line.startswith('# Vertices:'):
                            current_model_vertices = int(line.split(':')[1].strip())
                        elif line.startswith('# Faces:'):
                            current_model_faces = int(line.split(':')[1].strip())
                            break
                else:  # File is a D00##.obj file
                    current_model_vertices = content.count("v")
                    current_model_faces = content.count("f")
                # Save shape data
                if len(shape_data) == 0:
                    shape_data = [
                        {'Shape Name': file,
                         'Shape Class': current_class,
                         'Number of Vertices': current_model_vertices,
                         'Number of Faces': current_model_faces,
                         'Type of Faces': 'Triangle',
                         '3D Bounding Box': bounding_box}
                    ]
                else:
                    shape_data.append({'Shape Name': file,
                                       'Shape Class': current_class,
                                       'Number of Vertices': current_model_vertices,
                                       'Number of Faces': current_model_faces,
                                       'Type of Faces': 'Triangle',
                                       '3D Bounding Box': bounding_box})
    # Path to the CSV file
    csv_file_path = "outputs/shape_data.csv"

    # CSV file headers
    headers = ['Shape Name', 'Shape Class', 'Number of Vertices', 'Number of Faces', 'Type of Faces', '3D Bounding Box']

    # Create CSV file and add headers
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(headers)

    # Append shape data to the CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=';')
        for shape in shape_data:
            writer.writerow(shape)


if __name__ == '__main__':
    main()
