wkdir = '/Users/alegouhy/tests/contours2mesh'
pol2mesh_dir = '/Users/alegouhy/dev/polygons_to_mesh'
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
os.chdir(wkdir)
sys.path.append(pol2mesh_dir)
sys.path.append(os.path.join(pol2mesh_dir, 'microdraw.py'))
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm

import polygons_to_mesh as pm
import utils
import register

#%%

contours_file = os.path.join(pol2mesh_dir, 'data/7_registered_contours.npz')
opts_list = pm.get_registered_contours(contours_file)
voxdim = np.array([0.1, 0.1, 1.25])

contours = []
decim = 1
get_simps = True
get_normals = False
npts = 100

for i, opts in enumerate(opts_list):
    if opts is None: continue

    pts, simps, normals = utils.opts_to_contour(opts, npts=npts,
                                                get_simps=get_simps, get_normals=get_normals)
    
    pts, mean, std = utils.normalise_pts([pts], mean=300, std=200)
    pts = pts[0]
    
    contour = pts, simps, normals
    contours.append(contour)
       
    utils.plot_contour(contour, col=[1,0,0])
    plt.title(i)
    plt.show()

# spacing = 0.07
spacing = 1
z_coords = np.arange(0,  spacing*len(opts_list), spacing)

opts_list = [contour[0] for contour in contours]
pts, simps = utils.bridge_contours(opts_list, z_coords)
pts[:,:2] = (pts[:,:2] * std + mean) * voxdim[:2]
pts[:,2] = pts[:,2] * voxdim[2]
poly_raw = utils.vtkpoly(pts, simps)
utils.write_vtkpoly(poly_raw, 'output/mesh_raw.vtp')


#%% rigid ICP

contours_rig = contours.copy()
niter = 20
transfo = 'rigid'
init = 'centroid'

reg = register.reg_linear(niter=niter, transfo=transfo, init=init, se=True, plot=False)

for i in range(len(contours)-1):
    
    ref_contour = contours_rig[i]
    mov_contour = contours[i+1]

    rig, moved_contour = reg.compute(ref_contour, mov_contour)
    contours_rig[i+1] = moved_contour
    
    plt.subplot(1,2,1)
    utils.plot_contour(contours[i], col=[1,0,0])
    utils.plot_contour(contours[i+1], col=[0,0,1])
    plt.subplot(1,2,2)
    utils.plot_contour(ref_contour, col=[1,0,0])
    utils.plot_contour(moved_contour, col=[0,0,1])
    plt.suptitle('rigid - ' + str(i))
    plt.show()

opts_rig = [utils.contours2opts(np.array(contour[0]), np.array(contour[1])) for contour in contours_rig]
pts_rig, simps_rig = utils.bridge_contours_2(opts_rig, z_coords, greedy=False)

pts_rig[:,:2] = (pts_rig[:,:2] * std + mean) * voxdim[:2]
pts_rig[:,2] = pts_rig[:,2] * voxdim[2]

poly_rig = utils.vtkpoly(pts_rig, simps_rig)
utils.write_vtkpoly(poly_rig, 'output/mesh_rig_3.vtp')


#%% affine dwich ICP

contours_aff = contours_rig.copy()
niter = 20
transfo = 'affine'
init = 'identity'

reg = register.reg_linear(niter=niter, transfo=transfo, init=init, se=True, plot=False)

for i in range(2, len(contours)-1):
    
    prev_contour = contours_aff[i-1]
    next_contour = contours_rig[i+1]
    mov_contour = contours_rig[i]

    aff_prev, _ = reg.compute(prev_contour, mov_contour)
    aff_next, _ = reg.compute(next_contour, mov_contour)
    
    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
    lin, trans = utils.aff_dehmgn(aff)
    
    mov_pts, mov_simps, _ = mov_contour
    moved_pts = mov_pts @ lin.T + trans
    moved_contour = moved_pts, mov_simps, None
    contours_aff[i] = moved_contour

    plt.subplot(1,2,1)
    utils.plot_contour(prev_contour, col=[1,0,0])
    utils.plot_contour(mov_contour, col=[0,0,1])
    plt.subplot(1,2,2)
    utils.plot_contour(prev_contour, col=[1,0,0])
    utils.plot_contour(moved_contour, col=[0,0,1])
    plt.suptitle('affine - ' + str(i))
    plt.show()

opts_aff = [utils.contours2opts(np.array(contour[0]), np.array(contour[1])) for contour in contours_aff]
pts_aff, simps_aff = utils.bridge_contours_2(opts_aff, z_coords, greedy=False)

pts_aff[:,:2] = (pts_aff[:,:2] * std + mean) * voxdim[:2]
pts_aff[:,2] = pts_aff[:,2] * voxdim[2]
poly_aff = utils.vtkpoly(pts_aff, simps_aff)
utils.write_vtkpoly(poly_aff, 'output/mesh_dwich-aff_3.vtp')


#%% quad dwich ICP

contours_quad = contours_aff.copy()
niter = 20
degree = 2
init = 'identity'

reg = register.reg_polynom(niter=niter, degree=degree, init=init, se=True, plot=False)

for i in range(2, len(contours)-1):
    
    prev_contour = contours_quad[i-1]
    next_contour = contours_aff[i+1]
    mov_contour = contours_aff[i]

    moved_contour_prev = reg.compute(prev_contour, mov_contour)
    moved_contour_next = reg.compute(next_contour, mov_contour)
    
    moved_pts_prev, _, _ = moved_contour_prev
    moved_pts_next, _, _ = moved_contour_next
    moved_pts = (moved_pts_prev + moved_pts_next) / 2
    moved_contour = moved_pts, mov_contour[1], mov_contour[2]
    contours_quad[i] = moved_contour

    plt.subplot(1,2,1)
    utils.plot_contour(prev_contour, col=[1,0,0])
    utils.plot_contour(mov_contour, col=[0,0,1])
    plt.subplot(1,2,2)
    utils.plot_contour(prev_contour, col=[1,0,0])
    utils.plot_contour(moved_contour, col=[0,0,1])
    plt.suptitle('quadratic - ' + str(i))
    plt.show()

opts_quad = [utils.contours2opts(np.array(contour[0]), np.array(contour[1])) for contour in contours_quad]
pts_quad, simps_quad = utils.bridge_contours_2(opts_quad, z_coords, greedy=False)

pts_quad[:,:2] = (pts_quad[:,:2] * std + mean) * voxdim[:2]
pts_quad[:,2] = pts_quad[:,2] * voxdim[2]
poly_quad = utils.vtkpoly(pts_quad, simps_quad)
utils.write_vtkpoly(poly_quad, 'output/mesh_dwich-quad_3.vtp')


pts_quad_smo = utils.z_smooth(pts_quad, simps_quad)
poly_quad_smo = utils.vtkpoly(pts_quad_smo, simps_quad)
# poly_quad_smo = utils.smooth_vtkpoly(poly_quad)
utils.write_vtkpoly(poly_quad_smo, 'output/mesh_dwich-quad_3_smooth.vtp')


#%%

contours_test = contours_quad.copy()

for i, contour in enumerate(contours_test):
    
    pts, simps, normals = contour
    print(contour)

    npts = pts.shape[0]
    pts = np.concatenate((pts, np.full((npts, 1), z_coords[i])), axis=1)
    pts[:,:2] = pts[:,:2] * std + mean
    pts = pts * voxdim
    contours_test[i] = pts, simps, None

pts_test, simps_test, _ = utils.concat_contours(contours_test)

utils.plot_mesh(pts_test, simps_test)

poly_test = utils.vtkpoly(pts_test, simps_test)
utils.write_vtkpoly(poly_test, 'output/mesh_test.vtp')
utils.render_vtkpoly([poly_test])

#%%

mesh_rto_file = "/Users/alegouhy/dev/polygons_to_mesh/data/mesh_smooth.ply"
poly_rto = utils.read_vtkpoly(mesh_rto_file)
utils.write_vtkpoly(poly_rto, 'output/mesh_rto.vtp')



