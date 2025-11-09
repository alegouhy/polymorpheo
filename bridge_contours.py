wkdir = '/Users/alegouhy/tests/contours2mesh/'
import sys
sys.path.append(wkdir)
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import matplotlib.pyplot as plt
import numpy as np
import utils

#%%

img1_file = wkdir + 'imgs/Y_trunk.png'
img2_file = wkdir + 'imgs/Y_branch.png'
img1 = utils.load_img(img1_file)
img2 = utils.load_img(img2_file)

c1 = utils.seg_to_contour(img1)
c2 = utils.seg_to_contour(img2)
z_coords = [0, 10]

utils.plot_contour(c1, xlim=[0,30], ylim=[0,15])
utils.plot_contour(c2, xlim=[0,30], ylim=[0,17], col=[0,0,1])
plt.show()


#%% method 1: me

c1 = utils.seg_to_contour(img1)
c2 = utils.seg_to_contour(img2)
utils.plot_contour(c1, xlim=[0,30], ylim=[0,15])
utils.plot_contour(c2, xlim=[0,30], ylim=[0,17], col=[0,0,1])
plt.show()



opts_list = [c1[0], c2[0]]

pts, simps = utils.bridge_contours(opts_list, z_coords)

poly_raw = utils.vtkpoly(pts, simps)
utils.write_vtkpoly(poly_raw, wkdir + 'output/mesh_Y.vtp')


#%% Rasterize + marching cube

import skimage

opts_1 = skimage.measure.find_contours(img1 > 0)
opts_1 = [opts[:,[1,0]] for opts in opts_1]
opts_2 = skimage.measure.find_contours(img2 > 0)
opts_2 = [opts[:,[1,0]] for opts in opts_2]
contours = opts_rig.copy() # [opts_1, opts_1, opts_2, opts_2]
for opt in opts_1:
    plt.plot(opt[:,0], opt[:,1], '.-')
# for opt in opts_2:
#     plt.plot(opt[:,0], opt[:,1], '.-')
# plt.axis('off'); plt.show()



def rasterize(contours, imshape=None):
    
    vol = np.zeros((*imshape, len(contours)), dtype=np.uint8)

    for c, contour in enumerate(contours):
        for opt in contour:
            img = skimage.draw.polygon2mask(imshape, opt)
            vol[:,:,c] = np.logical_or(vol[:,:,c], img)
        
    return vol


def contours2mesh(contours, spacing=[1,1], paired=False):
    
    spacing = np.array(spacing)
    
    all_pts = np.vstack([opt for contour in contours for opt in contour])
    mini = all_pts.min(axis=0)
    maxi = all_pts.max(axis=0)
    
    contours_res = []
    for c, contour in enumerate(contours):
        contour_res = []
        for p, opt in enumerate(contour):
            contour_res.append((opt - mini) * spacing + 1)
        contours_res.append(contour_res)

    imshape = ((maxi - mini) * spacing + 3).astype(int)
    
    vol = rasterize(contours_res, imshape)
    
    pts, simps, normals, _ = skimage.measure.marching_cubes(vol, level=0.5)

    pts[:,:2] = ((pts[:,:2] - 0.5) / spacing) + mini
    
    print(np.min(all_pts, axis=0), np.min(pts, axis=0))
    print(np.max(all_pts, axis=0), np.max(pts, axis=0))
    
    return pts, simps, normals


pts, simps, _ = contours2mesh(contours, spacing=[100]*2)
# pts, simps, _ = contours2mesh(contours, spacing=[1]*2)
utils.plot_mesh(pts, simps)



#%% method 2: TO DO

import skimage

opts_1 = skimage.measure.find_contours(img1 > 0)
opts_1 = [opts[:,[1,0]] for opts in opts_1]
opts_2 = skimage.measure.find_contours(img2 > 0)
opts_2 = [opts[:,[1,0]] for opts in opts_2]
opts = [opts_1, opts_1, opts_2, opts_2]
for opt in opts_1:
    plt.plot(opt[:,0], opt[:,1], '.-')
for opt in opts_2:
    plt.plot(opt[:,0], opt[:,1], '.-')
plt.axis('off'); plt.show()



for i in range(len(opts)-1):
        
    opts_1 = opts[i]
    opts_2 = opts[i+1]
    
    n1 = len(opts_1)
    n2 = len(opts_2)
    if n1 == 1 and n2 == 2:
        opts_1 = utils.splitfit_opts(opts_1, opts_2)
        n1 == 1
    elif n1 == 2 and n2 == 1:
        opts_2 = utils.splitfit_opts(opts_2, opts_1)
        n2 += 1
    elif n1 != n2:
        raise ValueError(f"Can't handle {n1} to {n2} branching yet")
    
    for k in range(n1):
        
        opts_1k = opts_1[k]
        opts_2k = opts_2[k]
        
        opts_1k = utils.cw(opts_1k)
        opts_2k = utils.cw(opts_2k)
        
        opts_2k = utils.phase_align_contours(opts_1k, opts_2k)
        
        
new_opts_1 = utils.splitfit_opts(opts_1, opts_2)

for opt in new_opts_1:
    plt.plot(opt[:,0], opt[:,1], '.-')
    # plt.plot([opt[0,0],opt[-1,0]], [opt[0,1],opt[-1,1]], '.-');     
    plt.show()
    
    
a = [new_opts_1[1], np.roll(opts_2[1][:-1,:], 36, axis=0)]
for opt in a:
    plt.plot(opt[:,0], opt[:,1], '.-')   
plt.show()
    
a[1] = utils.phase_align_contours(a[0], a[1], start=True)

for opt in a:
    plt.plot(opt[:,0], opt[:,1], '.-')  
plt.show()




# for k in range(len(new_opts_1)):
    
#     c1 = new_opts_1[k]
#     c2 = opts_2[k]
    
#     if shapely.geometry.LinearRing(c1).is_ccw == True:
#         c1 = np.flipud(c1)
#     if shapely.geometry.LinearRing(c1).is_ccw == True:
#         c2 = np.flipud(c2)
    
#     plt.plot(c1[:,0], c1[:,1], '.-')
#     plt.plot(c2[:,0], c2[:,1], '.-')
#     plt.show()
        
        
    
# dists = [np.min(utils.pts_sqdist(opts_1[0], opt), axis=1) for opt in opts_2]
# assign = np.argmin(np.stack(dists, axis=-1), axis=1)

# dist = np.sqrt(utils.pts_sqdist(pts2,pts1))
# plt.imshow(1 / dist, vmin=0, vmax=1)
# plt.show()




# def frechet_dist(contour1, contour2):

#     c1 = contour1.reshape(-1, 2)
#     c2 = contour2.reshape(-1, 2)
    
#     n, m = c1.shape[0], c2.shape[0]
    
#     def dist(i, j):
#         return np.linalg.norm(c1[i] - c2[j])
    
#     # Only keep previous and current row
#     prev_row = np.full(m, np.inf)
#     curr_row = np.full(m, np.inf)
    
#     # Initialize first row
#     prev_row[0] = dist(0, 0)
#     for j in range(1, m):
#         prev_row[j] = max(prev_row[j-1], dist(0, j))
    
#     # Process remaining rows
#     for i in range(1, n):
#         curr_row[0] = max(prev_row[0], dist(i, 0))
        
#         for j in range(1, m):
#             curr_row[j] = max(
#                 min(prev_row[j], curr_row[j-1], prev_row[j-1]),
#                 dist(i, j)
#             )
        
#         prev_row, curr_row = curr_row, prev_row
    
#     return prev_row[-1]

# a = frechet_dist(opts_1[0],opts_2[0])

# line1 = shapely.LineString(opts_1[0])
# line2 = shapely.LineString(opts_2[0])
# b = shapely.frechet_distance(line1, line2)

# print(a, b)

#%% method 3: vtk

import skimage
import vtk

opts_1 = skimage.measure.find_contours(img1 > 0)
opts_1 = [opts[:,[1,0]] for opts in opts_1]
opts_2 = skimage.measure.find_contours(img2 > 0)
opts_2 = [opts[:,[1,0]] for opts in opts_2]
contours = [opts_1, opts_2]

append = vtk.vtkAppendPolyData()

for z, contour in zip(z_coords, contours):
    points = vtk.vtkPoints()
    polygon = vtk.vtkPolygon()
    for i, (x, y) in enumerate(contour):
        points.InsertNextPoint(x, y, z)
        polygon.GetPointIds().InsertNextId(i)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.InsertNextCell(polygon.GetCellType(), polygon.GetPointIds())
    append.AddInputData(polydata)

append.Update()
ruled = vtk.vtkRuledSurfaceFilter()
ruled.SetInputConnection(append.GetOutputPort())
ruled.Update()

mesh = ruled.GetOutput()


utils.render_vtkpoly([surface])


#%% method 4: rto

import skimage
import polygons_to_mesh as pm

opts_1 = skimage.measure.find_contours(img1 > 0)
opts_1 = [opts[:,[1,0]] for opts in opts_1]
opts_2 = skimage.measure.find_contours(img2 > 0)
opts_2 = [opts[:,[1,0]] for opts in opts_2]
registered_contours = [opts_1, opts_1, opts_2, opts_2]

verts, registered_contours_index = pm.get_vertices(registered_contours)
all_corresp = pm.all_correspondances(registered_contours)
all_corresp_pairs = pm.all_correspondances_pairs(all_corresp, registered_contours)
print(f"Number of correspondance pairs: {len(all_corresp_pairs)}")
comps = pm.get_all_connected_components(all_corresp_pairs)


cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(comps)))
plt.figure(figsize=(15,3))
for ind, comp in enumerate(comps):
    color = colors[ind]
    pm.plot_connected_component(comp, color=color, comp_label=f"Component {ind}")
plt.legend(prop={'size': 6})
pm.plot_correspondances_nodes(all_corresp)
plt.show()


plt.figure()
for ind, comp in enumerate(comps):
    color = colors[ind]
    pm.plot_connected_component_slices(comp, registered_contours, color=color)
plt.axis('equal')
plt.show()

print("CONNECT POLYGONS")
all_tris = []
for cind in range(len(comps)):
    print(f"Component {cind}")
    comp = comps[cind]

    for index, (sl, reg_index_1, reg_index_2) in enumerate(comp):
        print(f"index: {index}")
        regs1, regs2 = pm.get_slice_regions(sl, registered_contours)
        r1 = regs1[reg_index_1]
        r2 = regs2[reg_index_2]

        if max(len(r1), len(r2)) < 50:
            freq = 0.25
        elif max(len(r1), len(r2)) < 100:
            freq = 0.2
        else:
            freq = 0.15

        coupling = pm.couple_region_pair(r1, r2, freq=freq, debug=True)
        tr = pm.ribbon_triangles(r1, r2, coupling, reg_index_1, reg_index_2, sl,
                    verts, registered_contours_index)
        if len(all_tris) == 0:
            all_tris = tr
        else:
            all_tris.extend(tr)


comp = comps[0]
index = 1
# comp = comps[1]
# index = 0
(sl, reg_index_1, reg_index_2) = comp[index]
regs1, regs2 = pm.get_slice_regions(sl, registered_contours)
plt.subplot(1,2,1)
for r1 in regs1:
    plt.plot(r1[:,0],r1[:,1])
plt.subplot(1,2,2)
for r2 in regs2:
    plt.plot(r2[:,0],r2[:,1])
plt.suptitle([index, sl])
plt.show()
r1 = regs1[reg_index_1]
r2 = regs2[reg_index_2]
freq = 0.2
coupling = pm.couple_region_pair(r1, r2, freq=freq, debug=True)
tr = pm.ribbon_triangles(r1, r2, coupling, reg_index_1, reg_index_2, sl,
            verts, registered_contours_index)

simps = tr.copy()
pts = verts.copy()
indpts = np.unique(simps)
mask = np.zeros(pts.shape[0], np.bool)
mask[indpts] = True
pts[~mask,:] = 0
pts[:,2] = (pts[:,2] - np.mean(pts[:,2])) * 50
utils.plot_mesh(pts, simps, 'pyvista')














