wkdir = '/Users/alegouhy/tests/contours2mesh'
pol2mesh_dir = '/Users/alegouhy/dev/polygons_to_mesh'
import os
import sys
os.chdir(wkdir)
sys.path.append(pol2mesh_dir)
sys.path.append(os.path.join(pol2mesh_dir, 'microdraw.py'))
import matplotlib.pyplot as plt
import numpy as np

import polygons_to_mesh as pm
import utils

#%%

contours_file = os.path.join(pol2mesh_dir, 'data/7_registered_contours.npz')
opts_list = pm.get_registered_contours(contours_file)

contours = []
decim = 1
get_simps = True
get_normals = True
npts = 100

for i, opts in enumerate(opts_list):
    if opts is None: continue

    pts, simps, normals = utils.opts_to_contour(opts, npts=npts,
                                                get_simps=get_simps, get_normals=get_normals)
    
    pts, mean, std =  utils.normalise_pts([pts], mean=300, std=200)
    
    contour = pts[0], simps, normals
    contours.append(contour)
       
    utils.plot_contour(contour, col=[1,0,0])
    plt.title(i)
    plt.show()

spacing = 0.07
z_coords = np.arange(0,  spacing*len(opts_list), spacing)

#%%

opts_list = [contour[0] for contour in contours]
pts, simps = utils.bridge_contours(opts_list, z_coords)
utils.plot_mesh(pts, simps, 'pyvista')

