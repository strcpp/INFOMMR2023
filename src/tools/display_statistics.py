import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import NearestNeighbors
import os

csv_file_path = os.path.join('src', 'tools', 'outputs', 'shape_data.csv')

def show_histogram(df, column_name, class_name):
    """
    Shows the histogram for a specific column
    :param df: Dataframe
    :param column_name: Column name from which the histogram is calculated
    :param class_name: Name of the class in case we want class-specific histograms
    """
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(df[column_name], bins=20, edgecolor='k', alpha=0.7, label=column_name)
    plt.axvline(df[column_name].mean(), color='r', linestyle='dashed', linewidth=2,
                label=f'Average {column_name}: {df[column_name].mean():.2f}')
    if class_name:
        plt.title(f'Histogram of {class_name}')
    else:
        plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Number of Shapes')
    plt.legend()

    # Add text labels for each bin with the exact number of samples
    for i in range(len(bins) - 1):
        bin_center = (bins[i] + bins[i + 1]) / 2
        plt.text(bin_center, n[i], f'{int(n[i])}', ha='center', va='bottom')

    plt.show()


def return_neighbors(number_of_outliers: int):
    """
    Returns the nearest neighbor to the average shape as well as the 5 farthest neighbors (outliers).
    :param number_of_outliers: Determines how many outliers will be returned
    """
    df = pd.read_csv(os.path.join(os.getcwd(), csv_file_path), delimiter=';')

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
    farthest_neighbors_indices = all_neighbors_indices[-number_of_outliers:]

    return df.iloc[nearest_neighbor_index], df.iloc[farthest_neighbors_indices]


def display_histograms():
    """
    Displays multiple histograms
    """
    df = pd.read_csv(os.path.join(os.getcwd(), csv_file_path), delimiter=';')

    # Display histogram for the number of vertices
    show_histogram(df, "Number of Vertices", None)

    # Display histogram for the number of faces
    show_histogram(df, "Number of Faces", None)

    unique_shape_classes = df['Shape Class'].unique()

    # Create histograms for each shape class
    for i, shape_class in enumerate(unique_shape_classes):
        class_df = df[df['Shape Class'] == shape_class]
        show_histogram(class_df, "Number of Vertices", shape_class)

        time.sleep(2)

        show_histogram(class_df, "Number of Faces", shape_class)
        time.sleep(2)


def return_bounding_box(model_name):
    df = pd.read_csv(os.path.join(os.getcwd(), csv_file_path), delimiter=';')
    shape = df[df['Shape Name'] == model_name]
    bounding_box = shape.iloc[0]['3D Bounding Box']
    # Remove unwanted characters like [ and ], then split by newline
    lines = bounding_box.replace('[', '').replace(']', '').strip().split('\r\n')
    
    # For each line, split by whitespace and convert to float
    bounding_box = [list(map(float, line.split())) for line in lines]

    # In case bounding box length is 1, for some reason
    if len(bounding_box) == 1:
        bounding_box = bounding_box[0]
        return [[bounding_box[0], bounding_box[1], bounding_box[2]], [bounding_box[3], bounding_box[4], bounding_box[5]]]

    return bounding_box
