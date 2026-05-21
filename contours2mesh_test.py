wkdir = '/Users/alegouhy/tests/contours2mesh'
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
outdir = '/Users/alegouhy/tests/contours2mesh/output'
# names = ['wmlh_registered_contours']
names = ['rh_registered_contours']

spacing = np.array([0.1, 0.1, 1.25])
npts = 100
npts_min = 5
icp_niter = 20
bidir = True
thr_conn = 0.4

mesher = c2m.bridge_contours(thr_conn=thr_conn, sealed=False)


#%% load contours

io = c2m.io(datadir, names, spacing, npts, npts_min)
polylines_raw = io.load(plot=False)

polylines_raw = polylines_raw[13:20]

#%% registration - rigid ICP


for method in [4]: # range(2, 6):
    
    
    transfo = 'rigid'
    print('method ' + str(method) + ', ' + transfo)
    init = 'centroid'
    reg = c2m.register_slices(method, transfo, init, bidir=bidir)
    polylines_rig = reg.compute(polylines_raw)
    # meshes_rig = mesher.compute(polylines_rig)
    # io.save(meshes_rig, outdir, suffix='rig_met-'+str(method))
    
    transfo = 'affine'
    print('method ' + str(method) + ', ' + transfo)
    init = 'identity'
    reg = c2m.register_slices(method, transfo, init, bidir=bidir)
    polylines_aff = reg.compute(polylines_rig)

    transfo = 'deformable'
    print('method ' + str(method) + ', ' + transfo)
    lr = 1e-2
    wreg = 5e-1
    int_steps = 16
    sigma = 1e-1
    icp_niter = 50
    fit_fun = energy.pointdist(agg='mean', l=2)
    regul_fun = energy.grad_disp(l=2)
    reg = c2m.register_slices(method, transfo, fit_fun=fit_fun, regul_fun=regul_fun, 
                              icp_niter=icp_niter, lr=lr, wreg=wreg, sigma=sigma, int_steps=int_steps, plot=False)
    polylines_defo = reg.compute(polylines_aff)
    meshes_defo = mesher.compute(polylines_defo)
    io.save(meshes_defo, outdir, suffix='defo_met-'+str(method))
