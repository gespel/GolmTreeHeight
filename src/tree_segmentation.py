"""
Baumsegmentierung aus ALS-Punktwolken.

Ablauf:
  1. CHM berechnen (DSM - DTM) aus der Originalwolke
  2. CHM glätten (Gauss), lokale Maxima = Baumspitzen als Watershed-Seeds
  3. Watershed-Segmentierung → jede Krone bekommt eine ID
  4. Zu kleine Segmente verwerfen (min_points)
  5. Punkte aus raw_trees-Datei den Segmenten zuordnen
  6. Jeden Baum als eigene LAS-Datei schreiben

Parameter-Richtwerte für mitteleuropäischen Mischwald:
  sigma        = 2.5  (CHM-Glättung in Pixeln, bei 0.5m Aufl. ≈ 1.25m)
  min_height   = 5.0  (Mindesthöhe eines Baumes in m)
  min_distance = 6.0  (Mindestabstand zweier Baumspitzen in m)
  min_points   = 20   (Mindestanzahl 3-D-Punkte pro Segment)
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import maximum_filter, gaussian_filter, watershed_ift

from tree_detection import make_chm
from reader import read_las, write_las


def find_tree_tops(chm_smooth, chm_raw, min_height=5.0, min_distance=6.0, resolution=1.0):
    """
    Lokale Maxima im geglätteten CHM = Baumspitzen.
    Höhenfilter wird auf dem Roh-CHM geprüft, damit echte Spitzen
    nicht durch Glättung unter den Schwellwert fallen.
    Gibt (row_array, col_array) zurück.
    """
    win = max(3, int(min_distance / resolution) | 1)
    is_local_max = maximum_filter(chm_smooth, size=win) == chm_smooth
    is_high = (chm_raw >= min_height) & ~np.isnan(chm_raw)
    return np.where(is_local_max & is_high)


def segment_trees_chm(chm, min_height=5.0, min_distance=6.0, resolution=1.0, sigma=2.5):
    """
    Watershed-Segmentierung des CHM.
    Gibt ein Label-Array zurück (0 = kein Baum, 1..N = Baum-ID).
    """
    chm_smooth = gaussian_filter(np.nan_to_num(chm), sigma=sigma)
    top_rows, top_cols = find_tree_tops(chm_smooth, chm, min_height, min_distance, resolution)

    if len(top_rows) == 0:
        return np.zeros(chm.shape, dtype=np.int32)

    # Marker-Array: -1 = Hintergrund (nie zuweisen), 0 = offen, >0 = Seed
    tree_mask = (chm >= min_height) & ~np.isnan(chm)
    markers = np.where(tree_mask, 0, -1).astype(np.int32)
    # IDs direkt per Vektorisierung setzen
    markers[top_rows, top_cols] = np.arange(1, len(top_rows) + 1, dtype=np.int32)

    # watershed_ift braucht uint; invertieren → Spitzen werden zu Minima
    inv = chm_smooth.max() - chm_smooth
    inv_u16 = (inv / inv.max() * 65534).astype(np.uint16)

    labels = watershed_ift(inv_u16, markers).astype(np.int32)
    labels[labels < 0] = 0
    labels[~tree_mask] = 0
    return labels


def assign_points(cloud, labels, x_min, y_min, resolution):
    """
    Weist jedem Punkt seine Baum-ID aus dem Label-Array zu.
    Gibt ein int32-Array zurück (0 = nicht zugeordnet).
    """
    rows_chm, cols_chm = labels.shape
    col_idx = ((cloud["x"] - x_min) / resolution).astype(int).clip(0, cols_chm - 1)
    row_idx = ((cloud["y"] - y_min) / resolution).astype(int).clip(0, rows_chm - 1)
    return labels[row_idx, col_idx]


def segment_and_write(full_cloud, tree_cloud, out_dir, stem,
                      resolution=0.5, min_height=5.0, min_distance=6.0,
                      sigma=2.5, min_points=20):
    """
    Segmentiert den Baum-Cloud und schreibt jeden Baum als eigene LAS-Datei.

    full_cloud   : Originalwolke (mit Bodenpunkten) für das CHM
    tree_cloud   : Gefilterte Baumpunkte (aus raw_trees_*.las)
    out_dir      : Ausgabeverzeichnis
    stem         : Namenspräfix, z. B. '2018'
    min_points   : Segmente mit weniger Punkten werden verworfen
    Gibt Anzahl geschriebener Bäume zurück.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chm, x_min, y_min, res = make_chm(full_cloud, resolution=resolution)
    labels = segment_trees_chm(chm, min_height, min_distance, res, sigma)
    tree_ids = assign_points(tree_cloud, labels, x_min, y_min, res)

    written = 0
    out_id = 1
    for raw_id in np.unique(tree_ids):
        if raw_id == 0:
            continue
        mask = tree_ids == raw_id
        if mask.sum() < min_points:
            continue
        subtree = {k: v[mask] for k, v in tree_cloud.items()}
        write_las(subtree, out_dir / f"tree_{stem}_{out_id:04d}.las")
        out_id += 1
        written += 1

    return written


if __name__ == "__main__":
    import shutil
    data_dir = Path("data")
    out_dir = data_dir / "trees"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    for als_path in sorted(data_dir.glob("ALS_*.las")):
        year = als_path.stem.split("_")[1]
        raw_path = data_dir / f"raw_trees_{year}.las"

        if not raw_path.exists():
            print(f"Kein raw_trees_{year}.las — übersprungen.")
            continue

        print(f"\n{als_path.name} + {raw_path.name} ...")
        full_cloud = read_las(als_path)
        tree_cloud = read_las(raw_path)

        n = segment_and_write(
            full_cloud, tree_cloud,
            out_dir=out_dir,
            stem=year,
            resolution=0.5,
            min_height=5.0,
            min_distance=6.0,
            sigma=2.5,
            min_points=20,
        )
        print(f"  {n} Bäume segmentiert → {out_dir}/tree_{year}_XXXX.las")
