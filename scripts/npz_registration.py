import logging
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from importlib.resources import files
from pathlib import Path

import numpy as np

import polymorpheo as c2m
import polymorpheo.energy as energy
from polymorpheo.log import configure_logging

configure_logging(level=logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

# create reports dir if it doesn't exist
REPORTS_DIR.mkdir(exist_ok=True)

datadir_path = str(files("polymorpheo.data").joinpath("sample_contours.npz"))

io = c2m.io(
    datadir=datadir_path.replace("/sample_contours.npz", ""),
    names=["sample_contours"],
    npts=10,
    npts_min=5,
    normalise=True,
)


polylines_raw = io.load()


# spacing = np.array([0.1, 0.1, 1.25])
spacing = np.array([1, 1, 12.5])
npts = 100
npts_min = 5
icp_niter = 20
bidir = True
thr_conn = [0.2, 0.5]
niter = 5

method = 4  # serial registration method

mesher = c2m.bridge_contours(thr_conn=thr_conn, sealed=True)


transfo = "rigid"
print("method " + str(method) + ", " + transfo)
init = "centroid"
reg = c2m.register_slices(
    method, transfo, init, bidir=bidir, plot=False, xlim=io.xlim, ylim=io.ylim
)
polylines_rig = reg.compute(polylines_raw)
meshes_rig = mesher.compute(polylines_rig)
io.save(meshes_rig, REPORTS_DIR, suffix="rig_met-" + str(method))
print("Saved rigid meshes in " + str(REPORTS_DIR))


transfo = "affine"
print("method " + str(method) + ", " + transfo)
init = "identity"
reg = c2m.register_slices(
    method, transfo, init, bidir=bidir, plot=False, xlim=io.xlim, ylim=io.ylim
)
polylines_aff = reg.compute(polylines_rig)
meshes_aff = mesher.compute(polylines_aff)
io.save(meshes_aff, REPORTS_DIR, suffix="aff_met-" + str(method))
print("Saved affine meshes in " + str(REPORTS_DIR))

# transfo = 'polynomial'
# degree = 2
# init = 'identity'
# print('method ' + str(method) + ', ' + transfo)
# reg = c2m.register_slices(method, transfo, init, degree=degree, bidir=bidir)
# polylines_quad = reg.compute(polylines_aff)
# meshes_quad = mesher.compute(polylines_quad)
# io.save(meshes_quad, outdir, suffix='quad_met-'+str(method))


transfo = "deformable"
print("method " + str(method) + ", " + transfo)
lr = 1e-2
wreg = 5e-1
int_steps = 16
sigma = 1e-1
icp_niter = 50
fit_fun = energy.pointdist(agg="mean", bidir=bidir)
# regul_fun = energy.alap(transfo='similarity', normtype='l2')
regul_fun = energy.grad_disp(l_norm=2)
reg = c2m.register_slices(
    method,
    transfo,
    fit_fun=fit_fun,
    regul_fun=regul_fun,
    niter=niter,
    icp_niter=icp_niter,
    lr=lr,
    wreg=wreg,
    sigma=sigma,
    int_steps=int_steps,
)
polylines_defo = reg.compute(polylines_aff)
meshes_defo = mesher.compute(polylines_defo)
io.save(meshes_defo, REPORTS_DIR, suffix="defo_met-" + str(method) + "-" + str(niter))
print("Saved deformable meshes in " + str(REPORTS_DIR))
