import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from importlib.resources import files

import numpy as np

import polymorpheo
import polymorpheo.energy as energy
from polymorpheo.log import configure_logging

configure_logging(level=logging.WARNING)

REPORTS_DIR = REPO_ROOT / "reports"

# create reports dir if it doesn't exist
REPORTS_DIR.mkdir(exist_ok=True)

datadir_path = str(files("polymorpheo.data").joinpath("sample_contours.npz"))

spacing = np.array([0.1, 0.1, 1.25])

npts = 100
npts_min = 5
icp_niter = 20
bidir = True
thr_conn = [0.2, 0.5]
niter = 1

method = 4  # serial registration method

io = polymorpheo.io(
    datadir=datadir_path.replace("/sample_contours.npz", ""),
    names=["sample_contours"],
    spacing=spacing,
    npts=npts,
    npts_min=npts_min,
)

mesher = polymorpheo.bridge_contours(thr_conn=thr_conn, sealed=True)

# %%

polylines_raw, z_coords = io.load()
meshes_raw = mesher.compute(polylines_raw, z_coords)
io.save(meshes_raw, REPORTS_DIR, suffix="raw")

transfo_type = "rigid"
print("method " + str(method) + ", " + transfo_type)
init = "centroid"
reg = polymorpheo.register_slices(method, transfo_type, init, bidir=bidir, plot=False, xlim=io.xlim, ylim=io.ylim)
polylines_rig, transfos_rig = reg.compute(polylines_raw)
meshes_rig = mesher.compute(polylines_rig, z_coords)
io.save(meshes_rig, REPORTS_DIR, suffix="rig_met-" + str(method))
print("Saved rigid meshes in " + str(REPORTS_DIR))


transfo_type = "affine"
print("method " + str(method) + ", " + transfo_type)
init = "identity"
reg = polymorpheo.register_slices(method, transfo_type, init, bidir=bidir, plot=False, xlim=io.xlim, ylim=io.ylim)
polylines_aff, transfos_aff = reg.compute(polylines_rig)
meshes_aff = mesher.compute(polylines_aff, z_coords)
io.save(meshes_aff, REPORTS_DIR, suffix="aff_met-" + str(method))
print("Saved affine meshes in " + str(REPORTS_DIR))


transfo_type = "deformable"
print("method " + str(method) + ", " + transfo_type)
lr = 1e-2
wreg = 5e-1
int_steps = 16
sigma = 1e-1
icp_niter = 50
fit_fun = energy.point2point(agg="mean", bidir=bidir)
# regul_fun = energy.alap(transfo='similarity', normtype='l2')
regul_fun = energy.grad_disp(l_norm=2)
reg = polymorpheo.register_slices(
    method,
    transfo_type,
    fit_fun=fit_fun,
    regul_fun=regul_fun,
    niter=niter,
    icp_niter=icp_niter,
    lr=lr,
    wreg=wreg,
    sigma=sigma,
    int_steps=int_steps,
)
polylines_defo, transfos_defo = reg.compute(polylines_aff)
meshes_defo = mesher.compute(polylines_defo, z_coords)
io.save(meshes_defo, REPORTS_DIR, suffix="defo_met-" + str(method) + "-" + str(niter))
print("Saved deformable meshes in " + str(REPORTS_DIR))



#%% test compo transfo

import jax.numpy as jnp
import polymorpheo.transfo as transfo_ops

polylines_chain = []
for i in range(len(polylines_raw)):
    q = jnp.array(polylines_raw[i][0])

    after_rig  = transfo_ops.apply_transfo_chain(transfos_rig[i], q)
    after_aff  = transfo_ops.apply_transfo_chain(transfos_aff[i], after_rig)
    after_defo = transfo_ops.apply_transfo_chain(transfos_defo[i], after_aff)

    err_rig  = float(jnp.max(jnp.abs(after_rig  - jnp.array(polylines_rig[i][0]))))
    err_aff  = float(jnp.max(jnp.abs(after_aff  - jnp.array(polylines_aff[i][0]))))
    err_defo = float(jnp.max(jnp.abs(after_defo - jnp.array(polylines_defo[i][0]))))
    print(f"slice {i:3d}: err_rig={err_rig:.2e}  err_aff={err_aff:.2e}  err_defo={err_defo:.2e}")

    _, simps, _, labs = polylines_raw[i]
    polylines_chain.append((np.array(after_defo), simps, None, labs))

meshes_chain = mesher.compute(polylines_chain, z_coords)
io.save(meshes_chain, REPORTS_DIR, suffix="chain_met-" + str(method))
print("Saved chain meshes in " + str(REPORTS_DIR))

#%% test invert transfo

for i in range(len(polylines_raw)):
    q = jnp.array(polylines_raw[i][0])
    chain = transfos_rig[i] + transfos_aff[i] + transfos_defo[i]

    forward = transfo_ops.apply_transfo_chain(chain, q)
    back    = transfo_ops.apply_transfo_chain(chain, forward, invert=True)

    err = float(jnp.max(jnp.abs(back - q)))
    print(f"slice {i:3d}: err_inv={err:.2e}")


polylines_inv = []
for i in range(len(polylines_defo)):
    q = jnp.array(polylines_defo[i][0])
    chain = transfos_rig[i] + transfos_aff[i] + transfos_defo[i]
    back = transfo_ops.apply_transfo_chain(chain, q, invert=True)
    _, simps, _, labs = polylines_defo[i]
    polylines_inv.append((np.array(back), simps, None, labs))

meshes_inv = mesher.compute(polylines_inv, z_coords)
io.save(meshes_inv, REPORTS_DIR, suffix="inv_met-" + str(method))
print("Saved inverse meshes in " + str(REPORTS_DIR))
