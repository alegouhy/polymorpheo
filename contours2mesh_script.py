wkdir = '/Users/alegouhy/tests/ferret_atlas_reg'
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.chdir(wkdir)
import matplotlib.pyplot as plt
import numpy as np

import utils
import energy
import contours2mesh as c2m


#%%

datadir = '/Users/alegouhy/dev/polygons_to_mesh/data'
outdir = os.path.join(wkdir, 'output')
# names = ['wmlh_registered_contours', 'lh_registered_contours']
names = ['lh_registered_contours']

# spacing = np.array([0.1, 0.1, 1.25])
spacing = np.array([1, 1, 12.5])
npts = 100
npts_min = 5
icp_niter = 20
bidir = True
thr_conn = [0.2, 0.5]
niter = 5

method = 4              # serial registration method 

mesher = c2m.bridge_contours(thr_conn=thr_conn, sealed=True)


#%% load contours

io = c2m.io(datadir, names, spacing, npts, npts_min)
polylines_raw = io.load(plot=True)
 
# meshes_raw = mesher.compute(polylines_raw)
# io.save(meshes_raw, outdir, suffix='raw')

pts, simps = utils.polylines_2d_3d(polylines_raw, 2, 12.5*np.arange(len(polylines_raw)))
fig = utils.plot_obj(pts, simps, zlim=[0,1000], ylim=[0,600], xlim=[0,500])
fig.show()


#%% registration - rigid ICP

  
transfo = 'rigid'
print('method ' + str(method) + ', ' + transfo)
init = 'centroid'
reg = c2m.register_slices(method, transfo, init, bidir=bidir,
                          plot=False, xlim=io.xlim, ylim=io.ylim)
polylines_rig = reg.compute(polylines_raw)
# meshes_rig = mesher.compute(polylines_rig)
# io.save(meshes_rig, outdir, suffix='rig_met-'+str(method))


transfo = 'affine'
print('method ' + str(method) + ', ' + transfo)
init = 'identity'
reg = c2m.register_slices(method, transfo, init, bidir=bidir,
                          plot=False, xlim=io.xlim, ylim=io.ylim)
polylines_aff = reg.compute(polylines_rig)
# meshes_aff = mesher.compute(polylines_aff)
# io.save(meshes_aff, outdir, suffix='aff_met-'+str(method))


# transfo = 'polynomial'
# degree = 2
# init = 'identity'
# print('method ' + str(method) + ', ' + transfo)
# reg = c2m.register_slices(method, transfo, init, degree=degree, bidir=bidir)
# polylines_quad = reg.compute(polylines_aff)
# meshes_quad = mesher.compute(polylines_quad)
# io.save(meshes_quad, outdir, suffix='quad_met-'+str(method))


transfo = 'deformable'
print('method ' + str(method) + ', ' + transfo)
lr = 1e-2
wreg = 5e-1
int_steps = 16
sigma = 1e-1
icp_niter = 50
fit_fun = energy.pointdist(agg='mean', bidir=bidir)
# regul_fun = energy.alap(transfo='similarity', normtype='l2')
regul_fun = energy.grad_disp(l=2)
reg = c2m.register_slices(method, transfo, fit_fun=fit_fun, regul_fun=regul_fun, niter=niter,
                          icp_niter=icp_niter, lr=lr, wreg=wreg, sigma=sigma, int_steps=int_steps)
polylines_defo = reg.compute(polylines_aff)
meshes_defo = mesher.compute(polylines_defo)
#  io.save(meshes_defo, outdir, suffix='defo_met-'+str(method)+'-'+str(niter))
       
        