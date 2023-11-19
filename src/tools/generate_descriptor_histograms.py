from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import trimesh
import matplotlib.pyplot as plt

try:
    from display_statistics import return_shape_descriptors
except ModuleNotFoundError:
    try:
        from tools.display_statistics import return_shape_descriptors
    except ModuleNotFoundError:
        from src.tools.display_statistics import return_shape_descriptors


def load_model(path: tuple[str, str, str]) -> tuple[trimesh.Trimesh, str, str]:
    """
    Load model from path.
    :param path: Tuple containing the model's path, name and class.
    :return: Tuple containing the model's mesh, name and class.
    """
    mesh = trimesh.load_mesh(path[0])
    return mesh, path[1], path[2]


def main(show_plot=False):
    """
    Saves shape-property distribution plots for specific classes.
    :param show_plot: If True, the plots are also display.
    """
    normalized_model_path = os.path.join(os.path.dirname(__file__), '../../resources/models/Normalized')
    paths_to_load = []

    # Load Normalized models
    for root, dirs, files in os.walk(normalized_model_path):
        len_files = len(files)
        if len(files) > 0:
            for i in range(len_files):
                file = files[i]
                model_class = os.path.basename(os.path.normpath(root))
                path = os.path.join(normalized_model_path, model_class, file)
                paths_to_load.append((path, model_class, file))

    # Use multiprocessing to parallelize the loading
    with Pool(processes=cpu_count()) as pool:
        meshes = list(tqdm(pool.imap_unordered(load_model, paths_to_load)))

    all_model_names = {}
    all_meshes = {}
    for mesh in meshes:
        model_class = mesh[1]
        model_name = mesh[2]

        if model_class not in all_model_names:
            all_model_names[model_class] = []

        all_model_names[model_class].append(model_name)
        all_meshes[model_name] = mesh[0]

    # Split descriptors to descriptor-specific dictionaries
    all_descriptors = return_shape_descriptors(all_model_names, all_meshes)
    a3 = {}
    d1 = {}
    d2 = {}
    d3 = {}
    d4 = {}
    for _, descriptor in all_descriptors.items():
        if descriptor.model_class not in a3:
            a3[descriptor.model_class] = [descriptor.A3]
            d1[descriptor.model_class] = [descriptor.D1]
            d2[descriptor.model_class] = [descriptor.D2]
            d3[descriptor.model_class] = [descriptor.D3]
            d4[descriptor.model_class] = [descriptor.D4]
        else:
            a3[descriptor.model_class].append(descriptor.A3)
            d1[descriptor.model_class].append(descriptor.D1)
            d2[descriptor.model_class].append(descriptor.D2)
            d3[descriptor.model_class].append(descriptor.D3)
            d4[descriptor.model_class].append(descriptor.D4)

    # Run for each descriptor and class separately
    descriptors = ["A3", "D1", "D2", "D3", "D4"]
    for descriptor in descriptors:
        if descriptor == "A3":
            current_descriptor = a3
        elif descriptor == "D1":
            current_descriptor = d1
        elif descriptor == "D2":
            current_descriptor = d2
        elif descriptor == "D3":
            current_descriptor = d3
        else:
            current_descriptor = d4
        classes = ["Sign", "Wheel"]
        for c in classes:
            plt.figure()
            # Plot histograms and curves
            for i, series in enumerate(current_descriptor[c]):
                plt.plot(series, label=f'Series {i + 1}', color='black')

            # Add labels and legend
            plt.title(f'{descriptor} Descriptors for {c}')
            plt.xlabel(f'{len(current_descriptor[c])} {c}')
            plt.ylabel(descriptor)

            path = f"outputs/descriptors/{c}"
            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f"outputs/descriptors/{c}/{c}_{descriptor}")
            if show_plot:
                plt.show()


if __name__ == '__main__':
    main(show_plot=False)
