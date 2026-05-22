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

import polymorpheo.energy as energy
import polymorpheo.register as register
import polymorpheo.utils as utils

#%%

mov_file = REPO_ROOT / "imgs/mov_2.png"
ref_file = REPO_ROOT / "imgs/ref_2.png"

mov_img = utils.load_img(mov_file)
ref_img = utils.load_img(ref_file)

npts = 100
get_normals = False
bidir = True

mov_contour = utils.seg_to_contour(mov_img, npts=npts, get_normals=get_normals)
ref_contour = utils.seg_to_contour(ref_img, npts=npts, get_normals=get_normals)
mov_pts, mov_simps, mov_normals, mov_labs = mov_contour
ref_pts, ref_simps, ref_normals, ref_labs = ref_contour

[mov_pts, ref_pts], mean, std = utils.normalise_pts([mov_pts, ref_pts])
mov_contour = mov_pts, mov_simps, mov_normals, mov_labs
ref_contour = ref_pts, ref_simps, ref_normals, ref_labs

ref_pts_mu = np.mean(ref_pts, axis=0)
ref_pts_2 = (ref_pts - ref_pts_mu) * 1.2 + ref_pts_mu
ref_contour_2 = ref_pts_2, ref_simps, ref_normals, ref_labs
ref_contour = utils.concat_contours((ref_contour, ref_contour_2))

plt.figure(dpi=300)
plt.subplot(2,3,1)
utils.plot_contour(ref_contour)
utils.plot_contour(mov_contour)
plt.title('raw')


#%% rigid ICP

niter = 20
transfo = 'rigid'
init = 'centroid'

reg_rig = register.reg_linear(niter=niter, transfo=transfo, init=init, se=True, plot=False, bidir=bidir)

t = time.time()
rig, moved_contour_rig = reg_rig.compute(ref_contour, mov_contour)
<<<<<<< HEAD

=======
>>>>>>> main
print(f"rigid done in: {time.time()-t:.2f} s")

plt.subplot(2,3,2)
utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour_rig, col=[0,0,1])
plt.title('rigid')
# plt.show()

#%% affine ICP

niter = 20
transfo = 'similarity' # 'affine'

reg_aff = register.reg_linear(niter=niter, transfo=transfo, se=True, plot=False, bidir=bidir)

t = time.time()
aff, moved_contour_aff = reg_aff.compute(ref_contour, moved_contour_rig)
print(f"affine done in: {time.time()-t:.2f} s")

plt.subplot(2,3,3)
utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour_aff, col=[0,0,1])
plt.title('affine')
# plt.show()


#%% quadratic ICP

niter = 20
degree = 2

reg_poly = register.reg_polynom(niter=niter, degree=degree, init='identity', se=True, plot=False, bidir=bidir)

t = time.time()
moved_contour_quad = reg_poly.compute(ref_contour, moved_contour_aff)
print(f"poly done in: {time.time()-t:.2f} s")

plt.subplot(2,3,4)
utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour_quad, col=[0,0,1])
plt.title('quadratic')
# plt.show()


#%% cubic ICP

niter = 20
degree = 3

reg_poly = register.reg_polynom(niter=niter, degree=degree, init='identity', se=True, plot=False, bidir=bidir)

t = time.time()
moved_contour_cube = reg_poly.compute(ref_contour, moved_contour_aff)
print(f"poly done in: {time.time()-t:.2f} s")

plt.subplot(2,3,5)
utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour_cube, col=[0,0,1])
plt.title('cubic')

#%% non-rigid ICP

niter = 50
<<<<<<< HEAD
lr = [1e-2, 1e-3, 1e-3]
=======
lr = [1e-2] # , 1e-3, 1e-3]
>>>>>>> main
wreg = [1e-1, 1e-2, 1e-2]
int_steps = 16
sigma = [1e-1, 1e-2, 1e-3]
tol = 1e-6

fit_fun = energy.pointdist(agg='mean')
regul_fun = energy.grad_disp(l_norm=2)
moved_contour_defo = moved_contour_aff

losses = []
<<<<<<< HEAD
t = time.time()
for i in range(len(lr)):
    
    reg_defo = register.reg_deformable(niter=niter, fit_fun=fit_fun, regul_fun=regul_fun,
                                       lr=lr[i], wreg=wreg[i], sigma=sigma[i], int_steps=int_steps,
                                       tol=tol, plot=False)
    
    disp, moved_contour_defo, loss = reg_defo.compute(ref_contour, moved_contour_defo)
    
=======
reg_defos = []
t = time.time()
for i in range(len(lr)):

    reg_defo = register.reg_deformable(niter=niter, fit_fun=fit_fun, regul_fun=regul_fun,
                                       lr=lr[i], wreg=wreg[i], sigma=sigma[i], int_steps=int_steps,
                                       tol=tol, plot=False, normalise=True)

    disp, moved_contour_defo, loss = reg_defo.compute(ref_contour, moved_contour_defo)
    reg_defos.append(reg_defo)
>>>>>>> main
    losses.append(loss)

print(f"deformable done in: {time.time()-t:.2f} s")

plt.subplot(2,3,6)
utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour_defo, col=[0,0,1])
plt.title('deformable')


plt.tight_layout()
plt.show()

for i in range(len(lr)):
    plt.subplot(1,len(lr),i+1)
    plt.plot(losses[i])
    plt.suptitle('energy')
<<<<<<< HEAD
plt.show()
=======
plt.show()


#%% test transfo chain compo

import polymorpheo.transfo as transfo_ops

rig_lin, rig_trans = utils.aff_dehmgn(rig)
rig_transfo = transfo_ops.affine()
rig_transfo.set_params(rig_lin, rig_trans)

aff_lin, aff_trans = utils.aff_dehmgn(aff)
aff_transfo = transfo_ops.affine()
aff_transfo.set_params(aff_lin, aff_trans)


chain = [rig_transfo, aff_transfo] + [rd.polytransfo for rd in reg_defos]
moved_pts_chain = transfo_ops.apply_transfo_chain(chain, mov_pts)

import jax.numpy as jnp
print("max diff rig:", jnp.max(jnp.abs(moved_pts_chain - jnp.array(moved_contour_rig[0]))))

plt.figure()
# utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour((moved_pts_chain, mov_simps, None, mov_labs), col=[0,0,1])
utils.plot_contour(moved_contour_defo, col=[0,1,0])
plt.title('chain (blue) vs sequential (green)')
plt.show()




>>>>>>> main
