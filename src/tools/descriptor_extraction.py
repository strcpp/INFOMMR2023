import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast
from pymeshfix import MeshFix
import trimesh

output_dir1 = "tools/outputs/histograms/descriptors"
output_dir2 = "src/tools/outputs/histograms/descriptors"

np.random.seed(42)

SAMPLE_SIZE = 1000
BIN_SIZE = 10

def calculate_mesh_volume(mesh):
    # # Attempt to fix the mesh using pymeshfix
    # mfix = MeshFix(mesh.vertices, mesh.faces)
    # mfix.repair()
    # fixed_mesh = mfix.mesh

    # print(fixed_mesh)
    # # Convert fixed_mesh back to a trimesh.Trimesh object if necessary
    # mesh = trimesh.Trimesh(vertices=fixed_mesh.v, faces=fixed_mesh.f)

    # # Check if the mesh is watertight after repair
    if not mesh.is_watertight:
        mesh.fill_holes()

    # Calculate the volume using the watertight mesh
    reference_point = mesh.centroid
    volumes = []
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        tetra_volume = np.dot(reference_point - v0, np.cross(v1 - v0, v2 - v0)) / 6.0
        volumes.append(tetra_volume)

    volume = np.abs(sum(volumes))
    return volume


class ShapeDescriptors:
    def __init__(self, mesh, model_class, model_name, surface_area, compactness, rectangularity, diameter, convexity,
                 eccentricity, A3, D1, D2, D3, D4):
        self.mesh = mesh
        self.n_vertices = len(mesh.vertices)
        self.n_faces = len(mesh.faces)
        self.model_class = model_class
        self.model_name = model_name
        self.surface_area = surface_area
        self.surface_area_normalized = surface_area
        self.compactness = compactness
        self.compactness_normalized = compactness
        self.rectangularity = rectangularity
        self.rectangularity_normalized = rectangularity
        self.diameter = diameter
        self.diameter_normalized = diameter
        self.convexity = convexity
        self.convexity_normalized = convexity
        self.eccentricity = eccentricity
        self.eccentricity_normalized = eccentricity
        self.A3 = A3
        self.D1 = D1
        self.D2 = D2
        self.D3 = D3
        self.D4 = D4
        self.sample_size = SAMPLE_SIZE
        self.bin_size = BIN_SIZE

    @classmethod
    def from_csv_row(cls, row, mesh):

        model_class = row['Model Class'].item()
        model_name = row['Model Name'].item()

        surface_area = row['Surface Area'].iloc[0] if not pd.isnull(row['Surface Area'].iloc[0]) else 0.0

        compactness = row['Compactness'].iloc[0] if not pd.isnull(row['Compactness'].iloc[0]) else 0.0
        rectangularity = row['Rectangularity'].iloc[0] if not pd.isnull(row['Rectangularity'].iloc[0]) else 0.0
        diameter = row['Diameter'].iloc[0] if not pd.isnull(row['Diameter'].iloc[0]) else 0.0
        convexity = row['Convexity'].iloc[0] if not pd.isnull(row['Convexity'].iloc[0]) else 0.0
        eccentricity = row['Eccentricity'].iloc[0] if not pd.isnull(row['Eccentricity'].iloc[0]) else 0.0

        test = row['A3'].values[0]

        # Remove the square brackets and split the string into a list of strings
        data_str = test[1:-1]  # Remove square brackets
        numbers = data_str.split(', ')  # Split the string using ', ' as the separator

        # Convert the list of strings to a list of floats
        data = [float(num) for num in numbers]

        A3 = [float(num) for num in row['A3'].values[0][1:-1].split(', ')]
        D1 = [float(num) for num in row['D1'].values[0][1:-1].split(', ')]
        D2 = [float(num) for num in row['D2'].values[0][1:-1].split(', ')]
        D3 = [float(num) for num in row['D3'].values[0][1:-1].split(', ')]
        D4 = [float(num) for num in row['D4'].values[0][1:-1].split(', ')]

        return cls(
            mesh=mesh,
            model_class=model_class,
            model_name=model_name,
            surface_area=surface_area,
            compactness=compactness,
            rectangularity=rectangularity,
            diameter=diameter,
            convexity=convexity,
            eccentricity=eccentricity,
            A3=A3,
            D1=D1,
            D2=D2,
            D3=D3,
            D4=D4)

    @classmethod
    def from_mesh(cls, mesh, model_class, model_name):
        model_name, _ = os.path.splitext(model_name)
        surface_area = mesh.area

        volume = calculate_mesh_volume(mesh)

        compactness = cls.compute_compactness(mesh, volume)
        rectangularity = cls.compute_rectangularity(mesh, volume)
        diameter = cls.compute_diameter(mesh.convex_hull.vertices)
        convexity = cls.compute_convexity(mesh, volume)
        eccentricity = cls.compute_eccentricity(mesh)
        A3 = cls.compute_A3(mesh, SAMPLE_SIZE)
        D1 = cls.compute_D1(mesh, SAMPLE_SIZE)
        D2 = cls.compute_D2(mesh, SAMPLE_SIZE)
        D3 = cls.compute_D3(mesh, SAMPLE_SIZE)
        D4 = cls.compute_D4(mesh, SAMPLE_SIZE)

        return cls(
            mesh=mesh,
            model_class=model_class,
            model_name=model_name,
            surface_area=surface_area,
            compactness=compactness,
            rectangularity=rectangularity,
            diameter=diameter,
            convexity=convexity,
            eccentricity=eccentricity,
            A3=A3,
            D1=D1,
            D2=D2,
            D3=D3,
            D4=D4
        )

    def compute_compactness(mesh, volume):
        V = volume
        A = mesh.area
        return (A ** 3) / (V ** 2)

    def compute_rectangularity(mesh, volume):
        obb_volume = mesh.bounding_box_oriented.volume
        return volume / obb_volume

    @staticmethod
    @njit
    def compute_diameter(vertices):
        diameter = 0
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                diameter = max(diameter, dist)
        return diameter

    def compute_convexity(mesh, volume):
        return volume / mesh.convex_hull.volume

    def compute_eccentricity(mesh):
        covariance_matrix = np.cov(np.transpose(mesh.vertices))
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        return max(eigenvalues) / min(eigenvalues)

    def compute_A3(mesh, num_samples):
        angles = []
        vertices = mesh.vertices
        for _ in range(num_samples):
            A, B, C = vertices[np.random.choice(vertices.shape[0], 3, replace=False)]
            BA = A - B
            BC = C - B
            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angles.append(angle)

        histogram, bin_edges = np.histogram(angles, bins=BIN_SIZE, range=(0, np.pi))
        a3 = [x / np.sum(histogram) for x in histogram]
        return a3

    def save_A3_histogram_image(self):
        a3 = self.A3

        histogram, bin_edges = np.histogram(a3, bins=self.bin_size, range=(0, np.pi))

        fig, ax = plt.subplots(figsize=(10, 6))
        bin_edges = np.linspace(0, np.pi, len(histogram) + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, histogram, width=np.pi / len(histogram), align='center', edgecolor='black')
        ax.set_xlabel('Angle (radians)')
        ax.set_ylabel('Frequency')
        ax.set_title('Angle between 3 random vertices')
        ax.set_xlim(0, np.pi)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the figure directly to the desired path
        filename = f"A3_{self.model_class}_{self.model_name}.png"

        try:
            output_path = os.path.join(output_dir1, filename)
        except FileNotFoundError:
            output_path = os.path.join(output_dir2, filename)

        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D1(mesh, num_samples):
        barycenter = mesh.centroid

        distances = []
        vertices = mesh.vertices
        for _ in range(num_samples):
            # Sample a random vertex
            vertex = vertices[np.random.choice(vertices.shape[0])]
            # Compute the distance
            distance = np.linalg.norm(vertex - barycenter)
            distances.append(distance)
        histogram, bin_edges = np.histogram(distances, bins=BIN_SIZE)
        d1 = [x / np.sum(histogram) for x in histogram]
        return d1

    def save_D1_histogram_image(self):
        d1 = self.D1
        histogram, bin_edges = np.histogram(d1, bins=self.bin_size)
        self.D1 = [x / np.sum(histogram) for x in histogram]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(d1, bins=self.bin_size, edgecolor='black')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.set_title('Distancce between barycenter and random vertex')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the figure directly to the desired path
        filename = f"D1_{self.model_class}_{self.model_name}.png"
        try:
            output_path = os.path.join(output_dir1, filename)
        except FileNotFoundError:
            output_path = os.path.join(output_dir2, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D2(mesh, num_samples):
        distances = []
        vertices = mesh.vertices
        for _ in range(num_samples):
            # Sample two distinct random vertices
            vertex1, vertex2 = vertices[np.random.choice(vertices.shape[0], 2, replace=False)]

            # Compute the distance
            distance = np.linalg.norm(vertex1 - vertex2)
            distances.append(distance)
        histogram, bin_edges = np.histogram(distances, bins=BIN_SIZE)
        d2 = [x / np.sum(histogram) for x in histogram]
        return d2

    def save_D2_histogram_image(self):
        d2 = self.D2
        histogram, bin_edges = np.histogram(d2, bins=self.bin_size)
        self.D2 = [x / np.sum(histogram) for x in histogram]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(d2, bins=self.bin_size, edgecolor='black')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        ax.set_title('Distance between 2 random vertices')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the figure directly to the desired path
        filename = f"D2_{self.model_class}_{self.model_name}.png"
        try:
            output_path = os.path.join(output_dir1, filename)
        except FileNotFoundError:
            output_path = os.path.join(output_dir2, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D3(mesh, num_samples):
        areas = []
        vertices = mesh.vertices
        for _ in range(num_samples):
            # Sample three distinct random vertices
            A, B, C = vertices[np.random.choice(vertices.shape[0], 3, replace=False)]

            # Compute the lengths of the sides of the triangle
            a = round(np.linalg.norm(B - C), 3)
            b = round(np.linalg.norm(A - C), 3)
            c = round(np.linalg.norm(A - B), 3)

            # Compute the semi-perimeter
            s = round(((a + b + c) / 2), 3)

            # Compute the area using Heron's formula
            area = round(np.sqrt(np.abs(s * (s - a) * (s - b) * (s - c))), 3)
            areas.append(np.sqrt(area))
            try:
                histogram, bin_edges = np.histogram(areas, bins=BIN_SIZE)
            except ValueError:
                print(a)
                print(b)
                print(c)
                print(s)
                print(s * (s - a) * (s - b) * (s - c))
                print(area)
                raise ValueError
        d3 = [x / np.sum(histogram) for x in histogram]
        return d3

    def save_D3_histogram_image(self):
        d3 = self.D3

        histogram, bin_edges = np.histogram(d3, bins=self.bin_size)
        self.D3 = [x / np.sum(histogram) for x in histogram]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(d3, bins=self.bin_size, edgecolor='black')
        ax.set_xlabel('Square Root of Area')
        ax.set_ylabel('Frequency')
        ax.set_title('Square root of area of triangle given by 3 random vertices')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the figure directly to the desired path
        filename = f"D3_{self.model_class}_{self.model_name}.png"
        try:
            output_path = os.path.join(output_dir1, filename)
        except FileNotFoundError:
            output_path = os.path.join(output_dir2, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D4(mesh, num_samples):
        volumes = []
        vertices = mesh.vertices
        for _ in range(num_samples):
            # Sample four distinct random vertices
            A, B, C, D = vertices[np.random.choice(vertices.shape[0], 4, replace=False)]

            # Compute the volume of the tetrahedron
            AB = B - A
            AC = C - A
            AD = D - A
            volume = np.abs(np.dot(AB, np.cross(AC, AD))) / 6

            volumes.append(np.cbrt(volume))
        histogram, bin_edges = np.histogram(volumes, bins=BIN_SIZE)
        d4 = [x / np.sum(histogram) for x in histogram]
        return d4

    def save_D4_histogram_image(self):
        d4 = self.D4

        histogram, bin_edges = np.histogram(d4, bins=self.bin_size)
        self.D4 = [x / np.sum(histogram) for x in histogram]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(d4, bins=self.bin_size, edgecolor='black')
        ax.set_xlabel('Cube Root of Volume')
        ax.set_ylabel('Frequency')
        ax.set_title('Cube root of volume of tetrahedron formed by 4 random vertices')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        # Save the figure directly to the desired path
        filename = f"D4_{self.model_class}_{self.model_name}.png"
        try:
            output_path = os.path.join(output_dir1, filename)
        except FileNotFoundError:
            output_path = os.path.join(output_dir2, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def get_single_features(self):
        return [self.surface_area,
                self.compactness,
                self.rectangularity,
                self.diameter,
                self.convexity,
                self.eccentricity]

    def get_normalized_features(self):

        return_list = [self.surface_area_normalized * 0.025,
                       self.compactness_normalized * 0.025,
                       self.rectangularity_normalized * 0.025,
                       self.diameter_normalized * 0.025,
                       self.convexity_normalized * 0.025,
                       self.eccentricity_normalized * 0.025,
                       ]

        return_list.extend([x * 0.17 for x in self.A3])
        return_list.extend([x * 0.17 for x in self.D1])
        return_list.extend([x * 0.17 for x in self.D2])
        return_list.extend([x * 0.17 for x in self.D3])
        return_list.extend([x * 0.17 for x in self.D4])

        return return_list

    def get_normalized_features2(self):
        return_list = []
        return_list.extend(self.A3)
        return_list.extend(self.D1)
        return_list.extend(self.D2)
        return_list.extend(self.D3)
        return_list.extend(self.D4)

        return return_list

    def normalize_single_features(self, updated_features):
        self.surface_area_normalized = updated_features[0]
        self.compactness_normalized = updated_features[1]
        self.rectangularity_normalized = updated_features[2]
        self.diameter_normalized = updated_features[3]
        self.convexity_normalized = updated_features[4]
        self.eccentricity_normalized = updated_features[5]
