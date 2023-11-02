import numpy as np
from numba import njit
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

output_dir = "src/tools/outputs/histograms/descriptors"


class ShapeDescriptors:
    def __init__(self, mesh, model_class, model_name):
        self.mesh = mesh
        self.model_class = model_class
        self.model_name, _ = os.path.splitext(model_name)
        self.n_vertices = len(mesh.vertices)
        self.n_faces = len(mesh.faces)
        self.surface_area = mesh.area
        self.surface_area_normalized = self.surface_area
        self.compactness = self.compute_compactness()
        self.compactness_normalized = self.compactness
        self.rectangularity = self.compute_rectangularity()
        self.rectangularity_normalized = self.rectangularity
        self.diameter = ShapeDescriptors.compute_diameter(self.mesh.convex_hull.vertices)
        self.diameter_normalized = self.diameter
        self.convexity = self.compute_convexity()
        self.convexity_normalized = self.convexity
        self.eccentricity = self.compute_eccentricity()
        self.eccentricity_normalized = self.eccentricity
        self.sample_size = 1000
        self.bin_size = 10
        self.A3 = self.compute_A3(self.sample_size)
        self.D1 = self.compute_D1(self.sample_size)
        self.D2 = self.compute_D2(self.sample_size)
        self.D3 = self.compute_D3(self.sample_size)
        self.D4 = self.compute_D4(self.sample_size)

    def compute_compactness(self):
        V = self.mesh.volume
        A = self.mesh.area
        return (A ** 3) / (V ** 2)

    def compute_rectangularity(self):
        obb_volume = self.mesh.bounding_box_oriented.volume
        return self.mesh.volume / obb_volume

    @staticmethod
    @njit
    def compute_diameter(vertices):
        diameter = 0
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                dist = np.linalg.norm(vertices[i] - vertices[j])
                diameter = max(diameter, dist)
        return diameter

    def compute_convexity(self):
        return self.mesh.volume / self.mesh.convex_hull.volume

    def compute_eccentricity(self):
        covariance_matrix = np.cov(np.transpose(self.mesh.vertices))
        eigenvalues = np.linalg.eigvals(covariance_matrix)
        return max(eigenvalues) / min(eigenvalues)

    def compute_A3(self, num_samples):
        angles = []
        vertices = self.mesh.vertices
        for _ in range(num_samples):
            A, B, C = vertices[np.random.choice(vertices.shape[0], 3, replace=False)]
            BA = A - B
            BC = C - B
            cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angles.append(angle)
        self.A3 = angles
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

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the figure directly to the desired path
        filename = f"A3_{self.model_class}_{self.model_name}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D1(self, num_samples):
        barycenter = self.mesh.centroid

        distances = []
        vertices = self.mesh.vertices
        for _ in range(num_samples):
            # Sample a random vertex
            vertex = vertices[np.random.choice(vertices.shape[0])]
            # Compute the distance
            distance = np.linalg.norm(vertex - barycenter)
            distances.append(distance)
        self.D1 = distances
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

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the figure directly to the desired path
        filename = f"D1_{self.model_class}_{self.model_name}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D2(self, num_samples):
        distances = []
        vertices = self.mesh.vertices
        for _ in range(num_samples):
            # Sample two distinct random vertices
            vertex1, vertex2 = vertices[np.random.choice(vertices.shape[0], 2, replace=False)]

            # Compute the distance
            distance = np.linalg.norm(vertex1 - vertex2)
            distances.append(distance)
        self.D2 = distances
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

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the figure directly to the desired path
        filename = f"D2_{self.model_class}_{self.model_name}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D3(self, num_samples):
        areas = []
        vertices = self.mesh.vertices
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
        self.D3 = areas
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

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the figure directly to the desired path
        filename = f"D3_{self.model_class}_{self.model_name}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, format="png")
        plt.close(fig)

    def compute_D4(self, num_samples):
        volumes = []
        vertices = self.mesh.vertices
        for _ in range(num_samples):
            # Sample four distinct random vertices
            A, B, C, D = vertices[np.random.choice(vertices.shape[0], 4, replace=False)]

            # Compute the volume of the tetrahedron
            AB = B - A
            AC = C - A
            AD = D - A
            volume = np.abs(np.dot(AB, np.cross(AC, AD))) / 6

            volumes.append(np.cbrt(volume))
        self.D4 = volumes
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

        # Ensure the directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the figure directly to the desired path
        filename = f"D4_{self.model_class}_{self.model_name}.png"
        output_path = os.path.join(output_dir, filename)
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
        return [self.surface_area_normalized * 0.05,
                self.compactness_normalized * 0.05,
                self.rectangularity_normalized * 0.05,
                self.diameter_normalized * 0.05,
                self.convexity_normalized * 0.05,
                self.eccentricity_normalized * 0.05,
                self.A3[0] * 0.1,
                self.D1[0] * 0.15,
                self.D2[0] * 0.15,
                self.D3[0] * 0.15,
                self.D4[0] * 0.15
                ]

    def normalize_single_features(self, updated_features):
        self.surface_area_normalized = updated_features[0]
        self.compactness_normalized = updated_features[1]
        self.rectangularity_normalized = updated_features[2]
        self.diameter_normalized = updated_features[3]
        self.convexity_normalized = updated_features[4]
        self.eccentricity_normalized = updated_features[5]
