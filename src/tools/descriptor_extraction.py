import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast

output_dir1 = "tools/outputs/histograms/descriptors"
output_dir2 = "src/tools/outputs/histograms/descriptors"

np.random.seed(42)

SAMPLE_SIZE = 1000
BIN_SIZE = 10
class ShapeDescriptors:
    def __init__(self, mesh, model_class, model_name, surface_area, compactness, rectangularity, diameter, convexity, eccentricity, A3, D1, D2, D3, D4):
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

        model_class = row['Model Class']
        model_name = row['Model Name']

        surface_area = row['Surface Area'] if not row['Surface Area'].isnull() else 0.0
        compactness = row['Compactness'] if not row['Compactness'].isnull() else 0.0
        rectangularity = float(row['Rectangularity'].item()) if not row['Rectangularity'].isnull().item() else 0.0
        diameter = float(row['Diameter'].item()) if not row['Diameter'].isnull().item() else 0.0
        convexity = float(row['Convexity'].item()) if not row['Convexity'].isnull().item() else 0.0
        eccentricity = float(row['Eccentricity'].item()) if not row['Eccentricity'].isnull().item() else 0.0
        
        A3 = [float(x) for x in ast.literal_eval(row['A3'].iloc[0])] if pd.notna(row['A3'].iloc[0]) else []
        D1 = [float(x) for x in ast.literal_eval(row['D1'].iloc[0])] if pd.notna(row['D1'].iloc[0]) else []
        D2 = [float(x) for x in ast.literal_eval(row['D2'].iloc[0])] if pd.notna(row['D2'].iloc[0]) else []
        D3 = [float(x) for x in ast.literal_eval(row['D3'].iloc[0])] if pd.notna(row['D3'].iloc[0]) else []
        D4 = [float(x) for x in ast.literal_eval(row['D4'].iloc[0])] if pd.notna(row['D4'].iloc[0]) else []
        
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
        compactness = cls.compute_compactness(mesh)
        rectangularity = cls.compute_rectangularity(mesh)
        diameter = cls.compute_diameter(mesh.convex_hull.vertices)
        convexity = cls.compute_convexity(mesh)
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

    def compute_compactness(mesh):
        V = mesh.volume
        A = mesh.area
        return (A ** 3) / (V ** 2)

    def compute_rectangularity(mesh):
        obb_volume = mesh.bounding_box_oriented.volume
        return mesh.volume / obb_volume

    @staticmethod
    @njit
    def compute_diameter(vertices):
        diameter = 0
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                diameter = max(diameter, dist)
        return diameter

    def compute_convexity(mesh):
        return mesh.volume / mesh.convex_hull.volume

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
        return angles

    def save_A3_histogram_image(self):
        a3 = self.compute_A3(self.sample_size)

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
        D1 = distances
        return distances

    def save_D1_histogram_image(self):
        d1 = self.compute_D1(self.sample_size)

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
        D2 = distances
        return distances

    def save_D2_histogram_image(self):
        d2 = self.compute_D2(self.sample_size)

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
            a = np.linalg.norm(B - C)
            b = np.linalg.norm(A - C)
            c = np.linalg.norm(A - B)

            # Compute the semi-perimeter
            s = (a + b + c) / 2

            # Compute the area using Heron's formula
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            areas.append(np.sqrt(area))
        return areas

    def save_D3_histogram_image(self):
        d3 = self.compute_D3(self.sample_size)

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
        return volumes

    def save_D4_histogram_image(self):
        d4 = self.compute_D4(self.sample_size)

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

        return_list = [self.surface_area_normalized * 0.05,
                       self.compactness_normalized * 0.05,
                       self.rectangularity_normalized * 0.05,
                       self.diameter_normalized * 0.05,
                       self.convexity_normalized * 0.05,
                       self.eccentricity_normalized * 0.05,
                       ]

        return_list.extend([x * 0.1 for x in self.A3])
        return_list.extend([x * 0.15 for x in self.D1])
        return_list.extend([x * 0.15 for x in self.D2])
        return_list.extend([x * 0.15 for x in self.D3])
        return_list.extend([x * 0.15 for x in self.D4])

        return return_list

    def normalize_single_features(self, updated_features):
        self.surface_area_normalized = updated_features[0]
        self.compactness_normalized = updated_features[1]
        self.rectangularity_normalized = updated_features[2]
        self.diameter_normalized = updated_features[3]
        self.convexity_normalized = updated_features[4]
        self.eccentricity_normalized = updated_features[5]
