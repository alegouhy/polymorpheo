import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import time

import matplotlib.pyplot as plt
import numpy as np

import polymorpheo
import polymorpheo.energy as energy
import polymorpheo.register as register
import polymorpheo.utils as utils
from polymorpheo.log import configure_logging

configure_logging(level=logging.WARNING)

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

datadir_micro  = "/Users/alegouhy/dev/polygons_to_mesh/data"
name_micro     = "lh_registered_contours"
datadir_mri    = "/Users/alegouhy/data/polygon2mesh/p32-data"
name_mri       = "mesh-4_sym_left"

spacing_micro  = np.array([0.1, 0.1, 1.25])
npts_micro     = 100
npts_min_micro = 5
bidir          = True
reg_met        = 4

#%% load

micro_io = polymorpheo.io(datadir_micro, [name_micro], spacing_micro, npts_micro, npts_min_micro)
polylines_raw, z_coords = micro_io.load()

mri_poly           = utils.read_vtkpoly(os.path.join(datadir_mri, name_mri + ".ply"))
pts_mri, simps_mri = utils.vtkpoly2mesh(mri_poly)[:2]
normals_mri        = utils.normals_mesh(pts_mri, simps_mri)

#%% micro 2D slice registration

reg = polymorpheo.register_slices(reg_met, "rigid", "centroid", bidir=bidir, xlim=micro_io.xlim, ylim=micro_io.ylim)
polylines_rig, _ = reg.compute(polylines_raw)

reg = polymorpheo.register_slices(reg_met, "affine", "identity", bidir=bidir, xlim=micro_io.xlim, ylim=micro_io.ylim)
polylines_aff, _ = reg.compute(polylines_rig)

fit_fun   = energy.point2point(agg="mean", bidir=bidir)
regul_fun = energy.grad_disp(l_norm=2)
reg = polymorpheo.register_slices(reg_met, "deformable", fit_fun=fit_fun, regul_fun=regul_fun,
                                  niter=1, icp_niter=50, lr=1e-2, wreg=5e-1, sigma=1e-1, int_steps=16)
polylines_defo, _ = reg.compute(polylines_aff)

pts_micro, simps_micro = utils.polylines_2d_3d(polylines_defo, 2, z_coords)

#%% micro -> mri 3D registration

mesh_mri = pts_mri, None, normals_mri, None

pts_micro_init, _, _ = register.init_affcube(pts_mri, pts_micro)
print("cube init  -  dist:", utils.chamfer(pts_micro_init, pts_mri))
fig = utils.plot_obj(pts_mri, simps_mri)
fig = utils.plot_obj(pts_micro_init, simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
fig.show()
mesh_micro_init = pts_micro_init, simps_micro, None, None

reg_aff3d = register.reg_linear(niter=50, transfo="affine")
t = time.time()
_, mesh_micro_aff = reg_aff3d.compute(mesh_mri, mesh_micro_init)
pts_micro_aff = mesh_micro_aff[0]
print("affine  -  dist:", utils.chamfer(pts_micro_aff, pts_mri), ",  time:", time.time() - t)
fig = utils.plot_obj(pts_mri, simps_mri)
fig = utils.plot_obj(pts_micro_aff, simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
fig.show()

lrs         = [1e-2, 1e-2, 1e-3]
wregs       = [5e-1, 3e-1, 1e-1]
sigmas      = [5e-1, 1e-1, 5e-2]
cpts_ratios = [0.05, 0.1, 0.2]

mesh_micro_defo = mesh_micro_aff
for lr, wreg, sigma, cpts_ratio in zip(lrs, wregs, sigmas, cpts_ratios):
    fit_fun   = energy.point2plane(agg="mean", alpha=-2, scale=0.01, bidir=True)
    regul_fun = energy.alap(transfo="similarity", l_norm=2)
    regul_fun.set_neighs(mesh_micro_defo[1], mesh_micro_defo[0].shape[0])
    reg_defo  = register.reg_deformable(niter=50, fit_fun=fit_fun, regul_fun=regul_fun,
                                         lr=lr, wreg=wreg, sigma=sigma, int_steps=8, rk=2, cpts_ratio=cpts_ratio)
    t = time.time()
    _, mesh_micro_defo, loss = reg_defo.compute(mesh_mri, mesh_micro_defo)
    plt.plot(loss)
    plt.show()
    pts_micro_defo = mesh_micro_defo[0]
    print("deformable  -  dist:", utils.chamfer(pts_micro_defo, pts_mri), ",  time:", time.time() - t)
    fig = utils.plot_obj(pts_mri, simps_mri)
    fig = utils.plot_obj(pts_micro_defo, simps_micro, pts_col=(1, 0, 0), face_col=(1, 0.5, 0.5), fig=fig)
    fig.show()

