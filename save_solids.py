import numpy as np
import open3d as o3d
from solids import generate_sphere, generate_cube, generate_cone, generate_tetrahedron, generate_square_pyramid, generate_octahedron, visualize_point_cloud

def save_point_cloud(points, filename):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, cloud)

def main():
    shape = input("Wybierz figurę (sphere, cube, cone, tetrahedron, square_pyramid, octahedron): ").strip().lower()
    num_points = int(input("Podaj liczbę punktów (domyślnie 1000): "))
    filename = input("Podaj nazwę pliku do zapisu: ").strip()

    if shape == "sphere":
        points = generate_sphere(num_points)
    elif shape == "cube":
        points = generate_cube(num_points)
    elif shape == "cone":
        points = generate_cone(num_points)
    elif shape == "tetrahedron":
        points = generate_tetrahedron(num_points)
    elif shape == "square_pyramid":
        points = generate_square_pyramid(num_points)
    elif shape == "octahedron":
        points = generate_octahedron(num_points)
    else:
        print("Nieznana figura!")
        return

    save_point_cloud(points, filename)
    print(f"Chmura punktów dla {shape} została zapisana do pliku {filename}")

if __name__ == "__main__":
    main()