"""
Baumidentifikation aus ALS-Punktwolken.

Ablauf:
  1. DTM: Geländemodell aus Bodenpunkten (Klasse 2)
  2. DSM: Oberflächenmodell aus allen Punkten (höchstes Z pro Zelle)
  3. CHM: Canopy Height Model = DSM - DTM (normalisierte Höhe)
  4. Lokale Maxima im CHM → Baumspitzen
"""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, distance_transform_edt

CLASS_GROUND = 2
CLASS_BUILDING = 6


def make_raster(x, y, z, resolution=1.0, func=np.max,
                x_min=None, y_min=None, cols=None, rows=None):
    """
    Rasterisiert Punkte in ein 2D-Grid.
    Gibt (grid, x_min, y_min) zurück. Leere Zellen werden mit NaN gefüllt.
    """
    if x_min is None: x_min = x.min()
    if y_min is None: y_min = y.min()
    if cols is None: cols = int(np.ceil((x.max() - x_min) / resolution))
    if rows is None: rows = int(np.ceil((y.max() - y_min) / resolution))

    col_idx = ((x - x_min) / resolution).astype(int).clip(0, cols - 1)
    row_idx = ((y - y_min) / resolution).astype(int).clip(0, rows - 1)

    grid = np.full((rows, cols), np.nan)
    # Punkte nach Zelle gruppieren — schnell über np.lexsort + ufunc
    flat_idx = row_idx * cols + col_idx
    order = np.argsort(flat_idx)
    flat_sorted = flat_idx[order]
    z_sorted = z[order]

    unique_cells, first_occ = np.unique(flat_sorted, return_index=True)
    last_occ = np.append(first_occ[1:], len(flat_sorted))

    for cell, start, end in zip(unique_cells, first_occ, last_occ):
        grid[cell // cols, cell % cols] = func(z_sorted[start:end])

    return grid, x_min, y_min


def _fill_nan(grid):
    """Füllt NaN-Zellen mit dem nächsten gültigen Wert."""
    missing = np.isnan(grid)
    if missing.any():
        idx = distance_transform_edt(missing, return_distances=False, return_indices=True)
        grid = grid[tuple(idx)]
    return grid


def make_chm(cloud, resolution=1.0):
    """
    Berechnet das Canopy Height Model (CHM = DSM - DTM).
    Funktioniert mit und ohne Klassifikation:
      - DTM: Bodenpunkte (Klasse 2) falls vorhanden, sonst Minimum-Filter-Ansatz
      - Gebäude: Klasse 6 falls vorhanden, sonst Single-Return-Anteil > 80%
    Gibt (chm, x_min, y_min, resolution) zurück.
    """
    x, y, z = cloud["x"], cloud["y"], cloud["z"]
    cls = cloud["classification"]
    n_returns = cloud["number_of_returns"]

    # DSM: höchster Z-Wert pro Zelle
    dsm, x_min, y_min = make_raster(x, y, z, resolution, np.max)
    n_rows, n_cols = dsm.shape

    # --- DTM ---
    ground = cls == CLASS_GROUND
    if ground.sum() > n_rows * n_cols * 0.1:
        # Genug klassifizierte Bodenpunkte → direkt nutzen
        dtm, _, _ = make_raster(
            x[ground], y[ground], z[ground], resolution, np.min,
            x_min=x_min, y_min=y_min, cols=n_cols, rows=n_rows,
        )
        dtm = _fill_nan(dtm)
    else:
        # Der tiefste Punkt in einem ~10m-Fenster approximiert das Gelände
        dtm, _, _ = make_raster(x, y, z, resolution, np.min,
                                 x_min=x_min, y_min=y_min, cols=n_cols, rows=n_rows)
        dtm = _fill_nan(dtm)
        win = max(3, int(10 / resolution) | 1)
        dtm = minimum_filter(dtm, size=win)

    chm = np.clip(dsm - dtm, 0, None)

    # --- Gebäude maskieren ---
    if (cls == CLASS_BUILDING).sum() > 0:
        # Klassifikation verfügbar: Zellen mit Gebäudepunkten auf 0 setzen
        c_idx = ((x[cls == CLASS_BUILDING] - x_min) / resolution).astype(int).clip(0, n_cols - 1)
        r_idx = ((y[cls == CLASS_BUILDING] - y_min) / resolution).astype(int).clip(0, n_rows - 1)
        bld_mask = np.zeros((n_rows, n_cols), bool)
        bld_mask[r_idx, c_idx] = True
        chm[bld_mask] = 0
    else:
        # Kein Klassifikation: Zellen mit >80% Single-Returns sind wahrscheinlich Gebäude
        # Bäume erzeugen viele Mehrfach-Returns, flache Dächer fast nur Single-Returns
        single = (n_returns == 1).astype(float)
        sr_grid, _, _ = make_raster(x, y, single, resolution, np.mean,
                                     x_min=x_min, y_min=y_min, cols=n_cols, rows=n_rows)
        chm[(sr_grid > 0.8) & (chm > 2.0)] = 0

    return chm, x_min, y_min, resolution

def filter_tree_points(cloud, chm, x_min, y_min, resolution, min_height=2.0):
    """
    Filtert alle Punkte der Punktwolke, die zu einem Baum gehören.
    Ein Punkt gilt als Baumpunkt wenn die CHM-Höhe seiner Rasterzelle >= min_height.
    Gibt ein gefiltertes cloud-dict zurück.
    """
    x, y = cloud["x"], cloud["y"]
    rows_chm, cols_chm = chm.shape

    col_idx = ((x - x_min) / resolution).astype(int).clip(0, cols_chm - 1)
    row_idx = ((y - y_min) / resolution).astype(int).clip(0, rows_chm - 1)

    cell_height = chm[row_idx, col_idx]
    not_building = cloud["classification"] != CLASS_BUILDING
    mask = (cell_height >= min_height) & ~np.isnan(cell_height) & not_building

    return {k: v[mask] for k, v in cloud.items()}


if __name__ == "__main__":
    from pathlib import Path
    from reader import read_las, write_las

    for las_path in sorted(Path("data").glob("ALS_*.las")):
        year = las_path.stem.split("_")[1]
        out = f"data/raw_trees_{year}.las"
        print(f"\n{las_path.name}...")

        cloud = read_las(las_path)
        chm, x_min, y_min, res = make_chm(cloud, resolution=0.5)
        tree_cloud = filter_tree_points(cloud, chm, x_min, y_min, res, min_height=2.0)
        write_las(tree_cloud, out, classification=5)

        print(f"  Baumpunkte:  {len(tree_cloud['x']):,} von {len(cloud['x']):,}")
        print(f"  Gespeichert: {out}")
