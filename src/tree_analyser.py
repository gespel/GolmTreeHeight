from reader import read_las, write_las
from pathlib import Path
import numpy as np
import tqdm

def find_highest_tree():
    """
    Findet den höchsten Baum in der Punktwolke.
    Gibt die Baum-ID, Höhe und Koordinaten des höchsten Punktes zurück.
    """

    heighest_tree_id = None
    highest_tree_height = -np.inf
    highest_tree_coords = (None, None)

    for las_path in sorted(Path("data/filtered").glob("*.las")):
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
    for las_path in sorted(Path("data/filtered").glob(f"tree_{year}_*.las")):
        trees.append(las_path.stem)
    return trees

def find_tree_pair(tree_1):
    """
    Findet das nächstgelegene Baum-Paar zu einem gegebenen Baum zwischen 2018 und 2025.
    Gibt die IDs, Höhe und Koordinaten der beiden Bäume zurück.
    """

    tree_1_path = Path("data/filtered") / f"{tree_1}.las"
    if not tree_1_path.exists():
        raise FileNotFoundError(f"Baumdatei nicht gefunden: {tree_1_path}")

    cloud_1 = read_las(tree_1_path)
    x1, y1 = cloud_1["x"].mean(), cloud_1["y"].mean()

    closest_tree_id = None
    closest_distance = np.inf
    closest_tree_coords = (None, None)

    for las_path in sorted(Path("data/filtered").glob("*2025*.las")):
        if las_path.stem == tree_1:
            continue

        cloud_2 = read_las(las_path)
        x2, y2 = cloud_2["x"].mean(), cloud_2["y"].mean()

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if distance < closest_distance and distance < 1.5:  # Beispiel: nur Bäume innerhalb von 15 Metern berücksichtigen
            closest_distance = distance
            closest_tree_id = las_path.stem
            closest_tree_coords = (x2, y2)

    if closest_tree_id is not None:
        #print(f"\nNächstgelegener Baum zu {tree_1}: {closest_tree_id}")
        #print(f"Abstand: {closest_distance:.2f} m")
        #print(f"Koordinaten: x={closest_tree_coords[0]:.2f}, y={closest_tree_coords[1]:.2f}")
        return tree_1.split("_")[2], closest_tree_id.split("_")[2]  # Rückgabe der Baum-IDs ohne Jahr
    
def find_tree_with_highest_growth(tree_pairs):
    """ Gibt die drei Baum-Paare mit dem größten Wachstum aus und das Paar mit dem größten Wachstum zurück. """
    top_growth_pairs = []
    all_growth_values = []
    growth_data = []
    highest_growth_tree_pair = None
    highest_growth_value = -np.inf
    for tree_pair in tree_pairs:
        tree_1_id, tree_2_id = tree_pair
        tree_1_path = Path("data/filtered") / f"tree_2018_{tree_1_id}.las"
        tree_2_path = Path("data/filtered") / f"tree_2025_{tree_2_id}.las"

        if not tree_1_path.exists() or not tree_2_path.exists():
            print(f"Baumdatei nicht gefunden: {tree_1_path} oder {tree_2_path}")
            continue

        cloud_1 = read_las(tree_1_path)
        cloud_2 = read_las(tree_2_path)

        height_tree_1 = cloud_1["z"].max()
        height_tree_2 = cloud_2["z"].max()

        print(f"\nBaum-Paar: {tree_1_id} -> {tree_2_id}")
        print(f"Höhe Baum 1 ({tree_1_id}): {height_tree_1:.2f} m")
        print(f"Höhe Baum 2 ({tree_2_id}): {height_tree_2:.2f} m")

        growth_value = height_tree_2 - height_tree_1
        all_growth_values.append(growth_value)
        growth_data.append((tree_1_id, tree_2_id, growth_value))
        if growth_value > highest_growth_value:
            highest_growth_value = growth_value
            highest_growth_tree_pair = (tree_1_id, tree_2_id)

    print(f"\nDurchschnittliches Wachstum aller Baum-Paare: {np.mean(all_growth_values):.2f} m")

    # Top 3 Bäume mit dem größten Wachstum ausgeben
    sorted_trees = sorted(growth_data, key=lambda x: x[2], reverse=True)
    print("\n=== Top 3 Bäume mit dem größten Wachstum ===")
    for i, (tree_1_id, tree_2_id, growth) in enumerate(sorted_trees[:3], 1):
        print(f"{i}. Baum-Paar: {tree_1_id} -> {tree_2_id}, Wachstum: {growth:.2f} m")
        with open(f"data/growth_{i}.txt", "w") as f:
            f.write(f"Baum-Paar: {tree_1_id} -> {tree_2_id}, Wachstum: {growth:.2f} m\n")

    if highest_growth_tree_pair is not None:
        print(f"\nBaum mit dem größten Wachstum: {highest_growth_tree_pair[0]} -> {highest_growth_tree_pair[1]}")
        print(f"Wachstumswert: {highest_growth_value:.2f} m")
        with open("data/highest_growth.txt", "w") as f:
            f.write(f"Baum mit dem größten Wachstum: {highest_growth_tree_pair[0]} -> {highest_growth_tree_pair[1]}\n")
            f.write(f"Wachstumswert: {highest_growth_value:.2f} m\n")
        return highest_growth_tree_pair, highest_growth_value
    

if __name__ == "__main__":
    trees_2018 = get_trees_in_year(2018)
    
    #find_highest_tree()
    #find_tree_pair()
    tree_pairs = []
    for tree_2018 in tqdm.tqdm(trees_2018):
        pair = find_tree_pair(tree_2018)
        if pair is not None:
            tree_pairs.append(pair)

    
    find_tree_with_highest_growth(tree_pairs)