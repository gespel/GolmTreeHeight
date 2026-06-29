from reader import read_las, write_las
from pathlib import Path
import numpy as np

def find_highest_tree():
    """
    Findet den höchsten Baum in der Punktwolke.
    Gibt die Baum-ID, Höhe und Koordinaten des höchsten Punktes zurück.
    """

    heighest_tree_id = None
    highest_tree_height = -np.inf
    highest_tree_coords = (None, None)

    for las_path in sorted(Path("data/trees").glob("*.las")):
        year = las_path.stem.split("_")[1]
        print(f"{las_path.name}...")

        cloud = read_las(las_path)

        cloud_height = cloud["z"].max()
        if cloud_height > highest_tree_height:
            highest_tree_height = cloud_height
            heighest_tree_id = las_path.stem
            idx = np.argmax(cloud["z"])
            highest_tree_coords = (cloud["x"][idx], cloud["y"][idx])

    print(f"\nHöchster Baum: {heighest_tree_id}")
    print(f"Höhe: {highest_tree_height:.2f} m")
    print(f"Koordinaten: x={highest_tree_coords[0]:.2f}, y={highest_tree_coords[1]:.2f}")
    return heighest_tree_id, highest_tree_height, highest_tree_coords

def get_trees_in_year(year):
    """
    Gibt eine Liste der Baum-IDs für ein bestimmtes Jahr zurück.
    """
    trees = []
    for las_path in sorted(Path("data/trees").glob(f"tree_{year}_*.las")):
        trees.append(las_path.stem)
    return trees

def find_tree_pair(tree_1):
    """
    Findet das nächstgelegene Baum-Paar zu einem gegebenen Baum.
    Gibt die IDs, Höhe und Koordinaten der beiden Bäume zurück.
    """

    tree_1_path = Path("data/trees") / f"{tree_1}.las"
    if not tree_1_path.exists():
        raise FileNotFoundError(f"Baumdatei nicht gefunden: {tree_1_path}")

    cloud_1 = read_las(tree_1_path)
    x1, y1 = cloud_1["x"].mean(), cloud_1["y"].mean()

    closest_tree_id = None
    closest_distance = np.inf
    closest_tree_coords = (None, None)

    for las_path in sorted(Path("data/trees").glob("*.las")):
        if las_path.stem == tree_1:
            continue

        cloud_2 = read_las(las_path)
        x2, y2 = cloud_2["x"].mean(), cloud_2["y"].mean()

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_tree_id = las_path.stem
            closest_tree_coords = (x2, y2)

    print(f"\nNächstgelegener Baum zu {tree_1}: {closest_tree_id}")
    print(f"Abstand: {closest_distance:.2f} m")
    print(f"Koordinaten: x={closest_tree_coords[0]:.2f}, y={closest_tree_coords[1]:.2f}")
    return closest_tree_id, closest_distance, closest_tree_coords
    
    

if __name__ == "__main__":
    trees_2018 = get_trees_in_year(2018)

    #find_highest_tree()
    #find_tree_pair()
    for tree in trees_2018:
        find_tree_pair(tree)
        