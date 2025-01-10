import numpy as np
from scipy.spatial import ConvexHull

def mean_distance_from_center(points):
    """
    Odchylenie średnie
    """
    distances = np.linalg.norm(points, axis=1)
    return np.mean(distances)

def std_distance_from_center(points):
    """
    Odchylenie standardowe
    """
    distances = np.linalg.norm(points, axis=1)
    return np.std(distances)

def bounding_box(points):
    """
    Oblicza rozmiar prostopadłościanu obejmującego wszystkie punkty.
    Zwraca różnicę między maksymalną a minimalną współrzędną dla każdej osi.
    """
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return max_coords - min_coords

def convex_hull_volume(points):
    """
    Oblicza objętość wypukłej otoczki chmury punktów? xd
    """
    hull = ConvexHull(points)
    return hull.volume

def skewness(points):
    """
    Oblicza skośność rozkładu odległości punktów od środka.
    """
    distances = np.linalg.norm(points, axis=1)
    return np.mean(((distances - np.mean(distances)) / np.std(distances))**3)

def kurtosis(points):
    """
    Oblicza kurtozę rozkładu odległości punktów od środka.
    """
    distances = np.linalg.norm(points, axis=1)
    return np.mean(((distances - np.mean(distances)) / np.std(distances))**4) - 3

def moment_of_inertia(points):
    """
    Oblicza moment bezwładności chmury punktów względem środka.
    """
    distances = np.linalg.norm(points, axis=1)
    return np.sum(distances**2)

def feature_vector(points):
    """
    Wyciąga zestaw cech z chmury punktów.
    Zwraca wektor cech: [średnia odległość, odchylenie standardowe, rozmiary bounding box, objętość convex hull, skośność, kurtoza, moment bezwładności].
    """
    mean_dist = mean_distance_from_center(points)
    std_dist = std_distance_from_center(points)
    bbox = bounding_box(points)
    volume = convex_hull_volume(points)
    skew = skewness(points)
    kurt = kurtosis(points)
    inertia = moment_of_inertia(points)
    return [mean_dist, std_dist, *bbox, volume, skew, kurt, inertia]
