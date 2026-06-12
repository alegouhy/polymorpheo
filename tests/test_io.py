from pathlib import Path

from polymorpheo.io import io

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_io_npz_load() -> None:
    datadir = FIXTURES_DIR
    names = ["contours"]
    io_obj = io(datadir=datadir, names=names, npts=10, npts_min=5)
    polylines, z_coords = io_obj.load()

    assert len(polylines) == 80
