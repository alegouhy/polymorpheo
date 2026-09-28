# Slice images and contours of a MicroDraw project.
import io
import json
import os
import re
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from urllib.request import urlopen

import certifi
import numpy as np
from PIL import Image
from scipy import ndimage

from polymorpheo.io_micro import load_contour

try:
    import microdraw as md
except ImportError as err:  # not on PyPI: pip install git+https://github.com/r03ert0/microdraw.py
    raise ImportError("fetch_micro needs microdraw.py, see https://github.com/r03ert0/microdraw.py"
                      ) from err

ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
Image.MAX_IMAGE_PIXELS = None

TOKEN = ""

# scale level 15 is full resolution, 28000x19500, halved at each step down
_DATASET = {}


def dataset(project):
    if project not in _DATASET:
        prj = md.download_project_definition(project, TOKEN)
        source = prj["files"]["list"][0]["source"]
        ds = md.download_dataset_definition(source)
        _DATASET[project] = [source, ds["project"], ds["numSlices"]]
    return _DATASET[project]


def fetch(url, tries=5):
    for k in range(tries):
        try:
            return urlopen(url, timeout=60).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def level(project, slice_index, scale_level):
    source, desc, _ = dataset(project)
    url = desc["tileSources"][slice_index]
    if urlparse(url).scheme == "":
        base = urlparse(source)
        url = f"{base.scheme}://{base.netloc}{url}"
    txt = fetch(url).decode()
    get = lambda k: re.findall(f'{k}="([^"]+)"', txt)[0]
    w, h = int(get("Width")), int(get("Height"))
    down = 2 ** (int(np.ceil(np.log2(max(w, h)))) - scale_level)
    return (url, get("Format"), int(get("TileSize")), int(get("Overlap")),
            int(np.ceil(w / down)), int(np.ceil(h / down)))


def fetch_meta(outdir, project):
    source, desc, _ = dataset(project)
    os.makedirs(outdir, exist_ok=True)
    path = f"{outdir}/{project}.json"
    with open(path, "w") as f:
        json.dump({"source": source, **desc}, f)
    print(path, flush=True)
    return desc


def blacken_background(im, tol=30, min_size=100):
    # The slide around the section is the one flat colour covering most of it, so its median is
    # that colour and what sits further than tol from it, summed over the channels, is section.
    # The two are far apart, the palest tissue still being some 80 away, so tol sits in the gap
    # between them rather than close to either. The compressed edge of the section rings into that
    # gap in specks too small to be tissue, so only what holds together over min_size is kept.
    flat = im.reshape(-1, im.shape[2]) if im.ndim == 3 else im.reshape(-1, 1)
    section = np.abs(im - np.median(flat, axis=0).astype(im.dtype)).sum(-1) > tol

    labels, _ = ndimage.label(section)
    sizes = np.bincount(labels.ravel())
    section = np.isin(labels, np.nonzero(sizes >= min_size)[0][1:])

    im = im.copy()
    im[~section] = 0

    return im


def fetch_img(outdir, project, scale_level, slice_index=None, rm_background=False,
              overwrite=False):
    if slice_index is None:
        return [fetch_img(outdir, project, scale_level, i, rm_background, overwrite)
                for i in range(dataset(project)[2])]

    # a slice already on disk is kept, a run of the series being some 40 tile requests each, so
    # what changes what is written, rm_background, needs overwrite to reach the ones already there
    path = f"{outdir}/slice_{slice_index:04d}.jpg"
    if os.path.exists(path) and not overwrite:
        return np.asarray(Image.open(path))

    # microdraw's get_slice_image assumes 256 px tiles with no overlap
    url, fmt, tile, ov, lw, lh = level(project, slice_index, scale_level)
    grid = [(r, c) for r in range(-(-lh // tile)) for c in range(-(-lw // tile))]
    get = lambda rc: np.asarray(Image.open(io.BytesIO(fetch(
        f"{url[:-4]}_files/{scale_level}/{rc[1]}_{rc[0]}.{fmt}"))))

    im = None
    with ThreadPoolExecutor(8) as ex:
        for (r, c), t in zip(grid, ex.map(get, grid)):
            if im is None:
                im = np.zeros((lh, lw) + t.shape[2:], t.dtype)
            x, y = c * tile, r * tile
            w, h = min(tile, lw - x), min(tile, lh - y)
            im[y:y + h, x:x + w] = t[ov * (r > 0):, ov * (c > 0):][:h, :w]

    if rm_background:
        im = blacken_background(im)

    os.makedirs(outdir, exist_ok=True)
    Image.fromarray(im).save(path, quality=90)
    print(path, flush=True)
    return im


def fetch_contour(outdir, project, scale_level, slice_index=None):
    if slice_index is None:
        return [fetch_contour(outdir, project, scale_level, i)
                for i in range(dataset(project)[2])]

    path = f"{outdir}/slice_{slice_index:04d}.npz"
    if os.path.exists(path):
        return load_contour(path)

    source, desc, _ = dataset(project)
    width = level(project, slice_index, scale_level)[4]
    # regions are keyed by slice name, not by position in tileSources
    regions = md.download_all_regions_from_dataset_slice(
        source, project, desc["names"][slice_index], TOKEN)

    # the converter returns bare polygons, so it is called region by region to keep the names
    polys, names = [], []
    for region in regions:
        for poly in md.convert_microdraw_contours_to_resampled_polygons([region], width):
            polys.append(poly)
            names.append(region["annotation"]["name"])

    # filled elementwise: np.array would build a uniform 3D array when the contours match in shape
    contours = np.empty(len(polys), dtype=object)
    contours[:] = polys

    os.makedirs(outdir, exist_ok=True)
    np.savez(path, contours=contours, names=np.array(names))
    print(path, flush=True)
    return load_contour(path)

