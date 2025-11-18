wkdir = '/Users/alegouhy/tests/contours2mesh/'
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
os.chdir(wkdir)
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import make_valid

import utils



#%%

# img1_file = wkdir + 'imgs/Y_trunk.png'
# img2_file = wkdir + 'imgs/Y_branch.png'
# img3_file = wkdir + 'imgs/Y_left.png'
# img4_file = wkdir + 'imgs/Y_right.png'
# img1 = utils.load_img(img1_file)
# img2 = utils.load_img(img2_file)
# img3 = utils.load_img(img3_file)
# img4 = utils.load_img(img4_file)

# c1 = utils.seg_to_contour(img1)
# c2 = utils.seg_to_contour(img2)
# z_coords = [0, 10]

# utils.plot_contour(c1, xlim=[0,30], ylim=[0,15])
# utils.plot_contour(c2, xlim=[0,30], ylim=[0,17], col=[0,0,1])
# plt.show()


# import skimage

# z_coords = np.arange(0, 150, 5)
# opts_1 = skimage.measure.find_contours(img1 > 0)
# opts_1 = [opts[:,[1,0]] for opts in opts_1]
# opts_2 = skimage.measure.find_contours(img2 > 0)
# opts_2 = [opts[:,[1,0]] for opts in opts_2]
# opts_3 = skimage.measure.find_contours(img3 > 0)
# opts_3 = [opts[:,[1,0]] for opts in opts_3]
# opts_4 = skimage.measure.find_contours(img4 > 0)
# opts_4 = [opts[:,[1,0]] for opts in opts_4]
# opts_list = [opts_1, opts_1, opts_2, opts_2, opts_1, opts_1, opts_2, opts_3, opts_3, opts_2, opts_2, opts_1, opts_4]
# for opt in opts_1:
#     plt.plot(opt[:,0], opt[:,1], '.-')
# for opt in opts_2:
#     plt.plot(opt[:,0], opt[:,1], '.-')
# plt.axis('off'); plt.show()


# pts, simps = utils.bridge_contours_2(opts_list, z_coords, greedy=False)

# utils.plot_mesh(pts, simps, 'pyvista')
        



#%%

opts_quad = np.load('opts_quad.npy', allow_pickle=True)

spacing = 1
z_coords = np.arange(0,  spacing*len(opts_quad), spacing)
std = 200
mean = 300
voxdim = np.array([0.1, 0.1, 1.25])


#%%

# pts_quad, simps_quad = utils.bridge_contours_2(opts_quad[67:69], z_coords[67:69], greedy=False, sealed=False, thr_conn=0.4)
pts_quad, simps_quad = utils.bridge_contours_2(opts_quad, z_coords, greedy=False, sealed=False, thr_conn=0.4)

pts_quad[:,:2] = (pts_quad[:,:2] * std + mean) * voxdim[:2]
pts_quad[:,2] = pts_quad[:,2] * voxdim[2]

poly_quad = utils.vtkpoly(pts_quad, simps_quad)
utils.write_vtkpoly(poly_quad, 'output/mesh_dwich-quad_3.vtp')
utils.write_vtkpoly(poly_quad, 'output/mesh_dwich-quad_3.obj')

poly_quad_smo = utils.smooth_vtkpoly(poly_quad)
utils.write_vtkpoly(poly_quad_smo, 'output/mesh_dwich-quad_3_smooth.vtp')

