from pathlib import Path

import numpy as np

from polymorpheo.io_micro import io_micro

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_io_npz_load() -> None:
    datadir = FIXTURES_DIR
    names = ["contours"]
    io_obj = io_micro(datadir=datadir, names=names, npts=10, npts_min=5)
    polylines, z_coords = io_obj.load()

    assert len(polylines) == 80


def test_io_folder_load(tmp_path) -> None:
    # same fixture, written out one file per slice with two alternating region names
    series = np.load(FIXTURES_DIR / "contours.npz", allow_pickle=True)["registered_contours"]
    datadir = tmp_path / "contours_dir"
    datadir.mkdir()
    for z, opts in enumerate(series):
        opts = [] if opts is None else list(opts)
        contours = np.empty(len(opts), dtype=object)
        contours[:] = opts
        np.savez(datadir / f"slice_{z:04d}.npz", contours=contours,
                 names=np.array([f"region {i % 2}" for i in range(len(opts))]))

    io_obj = io_micro(datadir=tmp_path, names=["contours_dir"], npts=10, npts_min=5)
    polylines, _ = io_obj.load()

    assert len(polylines) == 80
    assert io_obj.nslice == len(series)
    assert io_obj.labels == ["region 0", "region 1"]

    labs = [np.unique(polyline[3]) for polyline in polylines]
    assert set(np.concatenate(labs)) <= {1, 2}
    # a region keeps its label across slices, and no contour closes into a self-edge
    assert all(1 in lab for lab in labs)
    simps = np.concatenate([polyline[1] for polyline in polylines])
    assert not (simps[:, 0] == simps[:, 1]).any()
