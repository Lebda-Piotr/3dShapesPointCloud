import numpy as np
import open3d as o3d

def generate_sphere(num_points=1000, radius=None):
    radius = radius if radius is not None else np.random.uniform(0.5, 2.0)  
    points = []
    for _ in range(num_points):
        phi = np.random.uniform(0, np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points.append([x, y, z])
    return np.array(points)

def generate_cube(num_points=1000, edge_length=None):
    edge_length = edge_length if edge_length is not None else np.random.uniform(0.5, 2.0)  
    points = []
    for _ in range(num_points):
        face = np.random.choice(6)
        if face == 0:  # front
            x = np.random.uniform(-edge_length/2, edge_length/2)
            y = np.random.uniform(-edge_length/2, edge_length/2)
            z = edge_length/2
        elif face == 1:  # back
            x = np.random.uniform(-edge_length/2, edge_length/2)
            y = np.random.uniform(-edge_length/2, edge_length/2)
            z = -edge_length/2
        elif face == 2:  # left
            x = -edge_length/2
            y = np.random.uniform(-edge_length/2, edge_length/2)
            z = np.random.uniform(-edge_length/2, edge_length/2)
        elif face == 3:  # right
            x = edge_length/2
            y = np.random.uniform(-edge_length/2, edge_length/2)
            z = np.random.uniform(-edge_length/2, edge_length/2)
        elif face == 4:  # top
            x = np.random.uniform(-edge_length/2, edge_length/2)
            y = edge_length/2
            z = np.random.uniform(-edge_length/2, edge_length/2)
        elif face == 5:  # bottom
            x = np.random.uniform(-edge_length/2, edge_length/2)
            y = -edge_length/2
            z = np.random.uniform(-edge_length/2, edge_length/2)
        points.append([x, y, z])
    return np.array(points)

def generate_cone(num_points=1000, radius=None, height=None):
    radius = radius if radius is not None else np.random.uniform(0.5, 2.0)
    height = height if height is not None else np.random.uniform(0.5, 4.0)
    points = []
    for _ in range(num_points):
        # Generowanie punktów na powierzchni bocznej stożka
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = height * (1 - r / radius)
        points.append([x, y, z])
    
    # Generowanie punktów na podstawie stożka
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0
        points.append([x, y, z])
    
    return np.array(points)

def generate_tetrahedron(num_points=1000, edge_length=None):
    edge_length = edge_length if edge_length is not None else np.random.uniform(0.5, 2.0)
    vertices = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
    vertices *= edge_length / np.sqrt(3)
    faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    points = []
    for _ in range(num_points):
        face = np.random.choice(4)
        a, b, c = vertices[faces[face]]
        u, v = np.random.uniform(size=2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        point = a*u + b*v + c*w
        points.append(point)
    return np.array(points)

def generate_square_pyramid(num_points=1000, base_length=None, height=None):
    base_length = base_length if base_length is not None else np.random.uniform(0.5, 4.0)
    height = height if height is not None else np.random.uniform(0.5, 4.0)
    base_points = np.array([
        [-base_length/2, -base_length/2, 0],
        [base_length/2, -base_length/2, 0],
        [base_length/2, base_length/2, 0],
        [-base_length/2, base_length/2, 0]
    ])
    apex = np.array([0, 0, height])
    points = []

    # Generowanie punktów na ścianach bocznych
    for _ in range(num_points // 2):
        face = np.random.choice(4)
        a, b = base_points[face], base_points[(face + 1) % 4]
        u, v = np.random.uniform(size=2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        point = a * (1 - u - v) + b * u + apex * v
        points.append(point)

    # Generowanie punktów na podstawie
    for _ in range(num_points // 2):
        a, b, c, d = base_points
        u, v = np.random.uniform(size=2)
        point = a * (1 - u) * (1 - v) + b * u * (1 - v) + c * u * v + d * (1 - u) * v
        points.append(point)
    return np.array(points)

def generate_octahedron(num_points=1000, edge_length=None):
    edge_length = edge_length if edge_length is not None else np.random.uniform(0.5, 2.0)
    vertices = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
    vertices *= edge_length / np.sqrt(2)
    faces = [[0, 2, 4], [0, 3, 4], [1, 2, 4], [1, 3, 4], [0, 2, 5], [0, 3, 5], [1, 2, 5], [1, 3, 5]]
    points = []
    for _ in range(num_points):
        face = np.random.choice(8)
        a, b, c = vertices[faces[face]]
        u, v = np.random.uniform(size=2)
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        point = a*u + b*v + c*w
        points.append(point)
    return np.array(points)


def visualize_point_cloud(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([cloud])

# Generowanie chmur punktów dla różnych brył z losowymi parametrami
sphere_points = generate_sphere()
cube_points = generate_cube()
cone_points = generate_cone()
tetrahedron_points = generate_tetrahedron()
square_pyramid_points = generate_square_pyramid()
octahedron_points = generate_octahedron()

# Wizualizacja brył
# visualize_point_cloud(sphere_points)
# visualize_point_cloud(cube_points)
# visualize_point_cloud(cone_points)
# visualize_point_cloud(tetrahedron_points)
# visualize_point_cloud(square_pyramid_points)
# visualize_point_cloud(octahedron_points)
