from pathlib import Path
import laspy
import numpy as np


def _fix_duplicate_dims(point_format):
    """Benennt doppelte Extra-Bytes-Felder um, damit numpy dtype() nicht wirft."""
    seen, dims = set(), list(point_format.dimensions)
    for i, d in enumerate(dims):
        name = d.name
        if name in seen:
            n = 2
            while f"{name}_{n}" in seen:
                n += 1
            name = f"{name}_{n}"
            dims[i] = d._replace(name=name)
        seen.add(name)
    point_format.dimensions = dims


def read_las(path):
    """Liest eine LAS-Datei und gibt ein dict mit x, y, z als numpy-Arrays zurück."""
    with laspy.open(path) as reader:
        _fix_duplicate_dims(reader.header.point_format)
        las = reader.read()
    return {
        "x": np.asarray(las.x),
        "y": np.asarray(las.y),
        "z": np.asarray(las.z),
    }


def get_z(cloud, x, y):
    """Gibt den Z-Wert des nächstgelegenen Punktes zur Koordinate (x, y)."""
    dx = cloud["x"] - x
    dy = cloud["y"] - y
    idx = np.argmin(dx * dx + dy * dy)
    return float(cloud["z"][idx])


def get_field_size(cloud):
    """Gibt (breite, hoehe) des Feldes in Metern zurück (x- und y-Ausdehnung)."""
    breite = float(cloud["x"].max() - cloud["x"].min())
    hoehe = float(cloud["y"].max() - cloud["y"].min())
    return breite, hoehe


def read_all(data_dir=None):
    """
    Liest alle .las-Dateien in data_dir (Standard: ../data).
    Gibt ein dict zurück: {dateiname: z_array}
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    return {p.stem: read_las(p) for p in sorted(Path(data_dir).glob("*.las"))}

if __name__ == "__main__":
    data = read_all()
    for name, cloud in data.items():
        z = cloud["z"]
        w, h = get_field_size(cloud)
        print(f"{name}: {len(z):,} Punkte, Z=[{z.min():.2f}, {z.max():.2f}] m, Feld={w:.0f}x{h:.0f} m")
