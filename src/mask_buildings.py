from pathlib import Path

import numpy as np
from PIL import Image
from reader import read_las
from scipy.ndimage import binary_dilation

CLASS_BUILDING: int = 6


def build_mask(width: int, height: int, save: bool = False) -> np.ndarray:
    """
    Lädt den ALS_2018 Datensatz, rastert ihn mit der gegebenen Auflösung und markiert alle Zellen,
    die Gebäude enthalten. Die Maske wird als PNG-Datei im data-Verzeichnis gespeichert.

    :param width: Breite der Maske in Zellen bzw. Pixeln.
    :param height: Höhe der Maske in Zellen bzw. Pixeln.
    :param save: Wenn True, wird die Maske als PNG-Datei im data-Verzeichnis gespeichert.
    :param grow: Anzahl der Nachbarzellen die ebenfalls als Gebäude markiert werden.
    :return: 2D-Array (height x width) mit True für Zellen, die Gebäude enthalten.
    """

    path: Path = Path("data/ALS_2018.las")

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    cloud = read_las(path)

    classes = cloud["classification"]

    if np.sum(classes == CLASS_BUILDING) == 0:
        raise Exception("Keine Gebäudeklassifikation in der Punktwolke vorhanden.")

    x = cloud["x"]
    y = cloud["y"]

    min_x = np.min(x)
    min_y = np.min(y)

    max_x = np.max(x)
    max_y = np.max(y)

    # Auflösung der Maske in Zellen / Pixel
    resolution_x: float = (max_x - min_x) / width
    resolution_y: float = (max_y - min_y) / height

    x_building = x[classes == CLASS_BUILDING]
    y_building = y[classes == CLASS_BUILDING]

    x_norm = x_building - min_x
    y_norm = y_building - min_y

    cols: int = (x_norm / resolution_x).astype(int).clip(0, width - 1)
    rows: int = (y_norm / resolution_y).astype(int).clip(0, height - 1)

    mask = np.zeros((height, width), bool)
    mask[rows, cols] = True

    if save:
        image = Image.fromarray(mask)
        image.save(f"data/mask_{width}x{height}.png")

    return mask


def main() -> None:
    build_mask(width=1920, height=1080, save=True)


if __name__ == "__main__":
    main()
