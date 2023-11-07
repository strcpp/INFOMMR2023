import time
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import tqdm
import warnings

try:
    from tools.descriptor_extraction import *
except ModuleNotFoundError:
    try:
        from descriptor_extraction import *
    except ModuleNotFoundError:
        from src.tools.descriptor_extraction import *

warnings.filterwarnings("ignore", category=RuntimeWarning)
csv_file_path = os.path.join('src', 'tools', 'outputs', 'shape_data.csv')
database_file_path = os.path.join('src', 'tools', 'outputs', 'database.csv')


def histogram(df, column_name, class_name, show):
    """
    Shows the histogram for a specific column
    :param df: Dataframe
    :param column_name: Column name from which the histogram is calculated
    :param class_name: Name of the class in case we want class-specific histograms
    :param show: Controls whether the histogram will be shown or not
    """
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(df[column_name], bins=35, edgecolor='k', alpha=0.7, label=column_name)
    plt.axvline(df[column_name].mean(), color='r', linestyle='dashed', linewidth=2,
                label=f'Average {column_name}: {df[column_name].mean():.2f}')
    if class_name:
        plt.title(f'Histogram of {class_name}')
    else:
        plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.xticks(bins, fontsize=5)
    plt.ylabel('Number of Shapes')
    plt.legend()

    # Add text labels for each bin with the exact number of samples
    for i in range(len(bins) - 1):
        bin_center = (bins[i] + bins[i + 1]) / 2
        plt.text(bin_center, n[i], f'{int(n[i])}', ha='center', va='bottom')

    if not os.path.exists("outputs/histograms/vertices/All"):
        os.makedirs("outputs/histograms/vertices/All")
    if not os.path.exists("outputs/histograms/faces/All"):
        os.makedirs("outputs/histograms/faces/All")

    if "Vertices" in column_name:
        if class_name:
            path = f"outputs/histograms/vertices/{class_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"outputs/histograms/vertices/{class_name}/vertices_{class_name}_normalized")
        else:
            plt.savefig(f"outputs/histograms/vertices/All/all_shapes_normalized")
    else:
        if class_name:
            path = f"outputs/histograms/faces/{class_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"outputs/histograms/faces/{class_name}/faces_{class_name}_normalized")
        else:
            plt.savefig(f"outputs/histograms/faces/All/all_shapes_normalized")

    if show:
        plt.show()


def return_neighbors():
    """
    Returns the nearest neighbor to the average shape as well as the 5 farthest neighbors (outliers).
    :return: Average shapes and all shapes
    """
    try:
        df = pd.read_csv(csv_file_path, delimiter=';')
    except FileNotFoundError:
        try:
            new_path = os.path.join('tools', 'outputs', 'shape_data.csv')
            df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')
        except FileNotFoundError:
            new_path = os.path.join('outputs', 'shape_data.csv')
            df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')

    average_vertices = df['Number of Vertices'].mean()
    average_faces = df['Number of Faces'].mean()

    x = df[['Number of Vertices', 'Number of Faces']].values

    # KNN where k is set to the number of shapes in the dataset in order to also find the outliers
    knn = NearestNeighbors(n_neighbors=len(df))
    knn.fit(x)

    # Create feature vector for the average shape
    average_shape = [[average_vertices, average_faces]]

    # Find the indices of all neighbors sorted by distance
    all_neighbors_indices = knn.kneighbors(average_shape, n_neighbors=len(df))[1][0]

    # Find nearest neighbor and 5 farthest neighbors
    nearest_neighbor_index = all_neighbors_indices[0]

    return df.iloc[nearest_neighbor_index], df.iloc[all_neighbors_indices]


def save_histograms(show_histogram):
    """
    Displays multiple histograms
    """
    try:
        df = pd.read_csv(csv_file_path, delimiter=';')
    except FileNotFoundError:
        try:
            new_path = os.path.join('tools', 'outputs', 'shape_data_normalized.csv')
            df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')
        except FileNotFoundError:
            new_path = os.path.join('outputs', 'shape_data_normalized.csv')
            df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')

    # Display histogram for the number of vertices
    histogram(df, "Number of Vertices", None, show_histogram)

    # Display histogram for the number of faces
    histogram(df, "Number of Faces", None, show_histogram)

    unique_shape_classes = df['Shape Class'].unique()

    # Create histograms for each shape class
    for i, shape_class in tqdm(enumerate(unique_shape_classes), desc="Saving Histograms", leave=False):
        class_df = df[df['Shape Class'] == shape_class]
        histogram(class_df, "Number of Vertices", shape_class, show_histogram)

        if show_histogram:
            time.sleep(2)

        histogram(class_df, "Number of Faces", shape_class, show_histogram)

        if show_histogram:
            time.sleep(2)


def return_bounding_box(model_name, mesh):
    if model_name:
        try:
            df = pd.read_csv(csv_file_path, delimiter=';')
        except FileNotFoundError:
            try:
                new_path = os.path.join('tools', 'outputs', 'shape_data.csv')
                df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')
            except FileNotFoundError:
                new_path = os.path.join('outputs', 'shape_data.csv')
                df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')

        shape = df[df['Shape Name'] == model_name]
        bounding_box = shape.iloc[0]['3D Bounding Box']
        # Remove unwanted characters like [ and ], then split by newline
        lines = bounding_box.replace('[', '').replace(']', '').strip().split('\r\n')

        # For each line, split by whitespace and convert to float
        bounding_box = [list(map(float, line.split())) for line in lines]
    else:
        bounding_box = mesh.bounds

    # In case bounding box length is 1, for some reason
    if len(bounding_box) == 1:
        bounding_box = bounding_box[0]
        return [[bounding_box[0], bounding_box[1], bounding_box[2]],
                [bounding_box[3], bounding_box[4], bounding_box[5]]]

    return bounding_box


def return_shape_descriptor(model_name, mesh):
    model_name = model_name.replace('.obj', '')
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), database_file_path))
    except FileNotFoundError:
        new_path = os.path.join('tools', 'outputs', 'database.csv')
        df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')

    row = df[df['Model Name'] == model_name]
    return ShapeDescriptors.from_csv_row(row, mesh)


def return_shape_descriptors(all_model_names, all_meshes):
    # Attempt to read the CSV file from the default path
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), database_file_path), delimiter=';')
    except FileNotFoundError:
        # If not found, try to read it from the alternative path
        new_path = os.path.join('tools', 'outputs', 'database.csv')
        df = pd.read_csv(os.path.join(os.getcwd(), new_path), delimiter=';')

    # Create a dictionary to hold the ShapeDescriptors, with model names as keys
    descriptors_map = {}

    # Loop over the items in all_model_names dictionary
    for model_name_key, model_names_list in all_model_names.items():
        # Loop through the list of model name strings
        for model_name_obj in model_names_list:
            # Standardize the model name by removing the '.obj' extension
            standardized_model_name = model_name_obj.replace('.obj', '')
            # Select the row in the dataframe that corresponds to the model name
            row = df[df['Model Name'] == standardized_model_name]
            # Retrieve the corresponding mesh object using the model name key
            mesh = all_meshes[model_name_obj]
            # Use the from_csv_row class method to create a ShapeDescriptor for each model name
            descriptors_map[model_name_obj] = ShapeDescriptors.from_csv_row(row, mesh)

    return descriptors_map
