wkdir = '/Users/alegouhy/tests/contours2mesh'
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.chdir(wkdir)
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm

import utils
import register


#%%

data_dir = '/Users/alegouhy/dev/polygons_to_mesh/data'
out_dir = '/Users/alegouhy/tests/contours2mesh/output'
# contours_name = ['7_registered_contours']
contours_name = ['wmlh_registered_contours', 'lh_registered_contours']
# contours_name = ['wmlh_registered_contours']
voxdim = np.array([0.1, 0.1, 1.25])
npts = 100
icp_niter = 20
bidir = True
# thr_conn = [0.2, 0.5]
thr_conn = [1/3] * 2

save_raw = True
save_rig = True
save_aff = True
save_quad = True


#%% load contours

contours_file = [os.path.join(data_dir, name + '.npz') for name in contours_name]
opts_lists = [np.load(file, allow_pickle=True)['registered_contours'] for file in contours_file]
nslice = len(opts_lists[0])
nlabs = len(contours_name)

pts_all = np.vstack([opt for opts_list in opts_lists for opts in opts_list if opts is not None for opt in opts])
pts_mu = np.mean(pts_all, axis=0)
pts_amp = np.max(np.abs(pts_all - pts_mu))

polylines_raw = []
for i in range(nslice):

    polyline = []
    for l in range(nlabs):
        opts = opts_lists[l][i]
        if opts is None: continue
        polyline_l = utils.opts_to_contour(opts, npts=npts, get_simps=True, lab=l+1)
        polyline.append(polyline_l)
    if len(polyline) == 0: continue

    pts, simps, _, labs = utils.concat_contours(polyline)
    
    pts = (pts - pts_mu) / pts_amp
    
    polyline = pts, simps, None, labs
    polylines_raw.append(polyline)
    
    utils.plot_contour(polyline)
    plt.title(str(i))
    plt.show()
  
nslice = len(polylines_raw)
midslice = int(nslice / 2)

#%% registration - rigid ICP

polylines_rig = polylines_raw.copy()
transfo = 'rigid'
init = 'centroid'

reg = register.reg_linear(niter=icp_niter, transfo=transfo, init=init, se=True, plot=False)

# for i in range(nslice-1):

#     ref_polyline = polylines_rig[i]
#     mov_polyline = polylines_raw[i+1]

#     rig, moved_polyline = reg.compute(ref_polyline, mov_polyline)
#     polylines_rig[i+1] = moved_polyline

#  - 

for i in range(midslice,nslice-1):
    
    ref_polyline = polylines_rig[i]
    mov_polyline = polylines_raw[i+1]

    rig, moved_polyline = reg.compute(ref_polyline, mov_polyline)
    polylines_rig[i+1] = moved_polyline
   
for i in reversed(range(1,midslice)):
    
    ref_polyline = polylines_rig[i]
    mov_polyline = polylines_raw[i-1]

    rig, moved_polyline = reg.compute(ref_polyline, mov_polyline)
    polylines_rig[i-1] = moved_polyline
    
    
#%% registration - affine dwich ICP

polylines_aff = polylines_rig.copy()
transfo = 'affine'
init = 'identity'

reg = register.reg_linear(niter=icp_niter, transfo=transfo, init=init, se=True, bidir=bidir, plot=False)


#  method 1

for i in range(1, nslice-1):
    
    prev_polyline = polylines_aff[i-1]
    next_polyline = polylines_rig[i+1]
    mov_polyline = polylines_rig[i]

    aff_prev, _ = reg.compute(prev_polyline, mov_polyline)
    aff_next, _ = reg.compute(next_polyline, mov_polyline)
    
    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
    lin, trans = utils.aff_dehmgn(aff)
    
    mov_pts, mov_simps, _, mov_labs = mov_polyline
    moved_pts = mov_pts @ lin.T + trans
    moved_polyline = moved_pts, mov_simps, None, mov_labs
    polylines_aff[i] = moved_polyline
    
# for i in reversed(range(1, nslice-1)):
    
#     prev_polyline = polylines_rig[i-1]
#     next_polyline = polylines_aff[i+1]
#     mov_polyline = polylines_aff[i]

#     aff_prev, _ = reg.compute(prev_polyline, mov_polyline)
#     aff_next, _ = reg.compute(next_polyline, mov_polyline)
    
#     aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
#     lin, trans = utils.aff_dehmgn(aff)
    
#     mov_pts, mov_simps, _ = mov_polyline
#     moved_pts = mov_pts @ lin.T + trans
#     moved_polyline = moved_pts, mov_simps, None
#     polylines_aff[i] = moved_polyline

# #  method 2

# for i in range(midslice+1,nslice-1):
        
#     prev_polyline = polylines_aff[i-1]
#     next_polyline = polylines_rig[i+1]
#     mov_polyline = polylines_rig[i]

#     aff_prev, _ = reg.compute(prev_polyline, mov_polyline)
#     aff_next, _ = reg.compute(next_polyline, mov_polyline)
    
#     aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
#     lin, trans = utils.aff_dehmgn(aff)
    
#     mov_pts, mov_simps, _, mov_labs = mov_polyline
#     moved_pts = mov_pts @ lin.T + trans
#     moved_polyline = moved_pts, mov_simps, None, mov_labs
#     polylines_aff[i] = moved_polyline
       
# for i in reversed(range(1,midslice)):
        
#     prev_polyline = polylines_aff[i+1]
#     next_polyline = polylines_rig[i-1]
#     mov_polyline = polylines_rig[i]

#     aff_prev, _ = reg.compute(prev_polyline, mov_polyline)
#     aff_next, _ = reg.compute(next_polyline, mov_polyline)
    
#     aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
#     lin, trans = utils.aff_dehmgn(aff)
    
#     mov_pts, mov_simps, _, mov_labs = mov_polyline
#     moved_pts = mov_pts @ lin.T + trans
#     moved_polyline = moved_pts, mov_simps, None, mov_labs
#     polylines_aff[i] = moved_polyline


# #  method 3

# polylines_aff0 = polylines_rig.copy()
    
# for _ in range(1):
    
#     for i in range(1, nslice, 2):
        
#         mov_polyline = polylines_aff0[i]
        
#         prev_polyline = polylines_aff0[i-1]
#         aff_prev, _ = reg.compute(prev_polyline, mov_polyline)
        
#         if i < nslice - 1:
#             next_polyline = polylines_aff0[i+1]
#             aff_next, _ = reg.compute(next_polyline, mov_polyline)
#         else:
#             aff_next = np.eye(3)
        
#         aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
#         lin, trans = utils.aff_dehmgn(aff)
        
#         mov_pts, mov_simps, _, mov_labs = mov_polyline
#         moved_pts = mov_pts @ lin.T + trans
#         moved_polyline = moved_pts, mov_simps, None, mov_labs
#         polylines_aff[i] = moved_polyline
     
#     polylines_aff0 = polylines_aff.copy()
    
#     for i in range(0, nslice, 2):
        
#         mov_polyline = polylines_aff0[i]
        
#         if i > 0:
#             prev_polyline = polylines_aff0[i-1]
#             aff_prev, _ = reg.compute(prev_polyline, mov_polyline)
#         else:
#             aff_prev = np.eye(3)
    
#         next_polyline = polylines_aff0[i+1]
#         aff_next, _ = reg.compute(next_polyline, mov_polyline)
    
#         aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
#         lin, trans = utils.aff_dehmgn(aff)
        
#         mov_pts, mov_simps, _, mov_labs = mov_polyline
#         moved_pts = mov_pts @ lin.T + trans
#         moved_polyline = moved_pts, mov_simps, None, mov_labs
#         polylines_aff[i] = moved_polyline
        
#     polylines_aff0 = polylines_aff.copy()
    
    
#  method 4

# polylines_aff0 = polylines_rig.copy()
    
# for _ in range(1):
    
#     for i in range(1, nslice, 2):
        
#         mov_polyline = polylines_aff0[i]
        
#         ref_polyline = [polylines_aff0[i-1]]
#         if i < nslice - 1:
#             ref_polyline += [polylines_aff0[i+1]]
            
#         ref_polyline = utils.concat_contours(ref_polyline)
#         aff, moved_polyline = reg.compute(ref_polyline, mov_polyline)

#         polylines_aff[i] = moved_polyline
     
#     polylines_aff0 = polylines_aff.copy()
    
#     for i in range(0, nslice, 2):
        
#         mov_polyline = polylines_aff0[i]
        
#         ref_polyline = [polylines_aff0[i+1]]
#         if i > 0:
#             ref_polyline += [polylines_aff0[i-1]]
        
#         ref_polyline = utils.concat_contours(ref_polyline)
#         aff, moved_polyline = reg.compute(ref_polyline, mov_polyline)

#         polylines_aff[i] = moved_polyline
        
#     polylines_aff0 = polylines_aff.copy()  
    

#  method 5

# polylines_aff0 = polylines_rig.copy()

# for _ in range(3):
    
#     for i in range(1, nslice-1):
        
#         mov_polyline = polylines_aff0[i]
        
#         ref_polyline = [polylines_aff0[i-1]]
#         ref_polyline += [polylines_aff0[i+1]]
            
#         ref_polyline = utils.concat_contours(ref_polyline)
#         aff, moved_polyline = reg.compute(ref_polyline, mov_polyline)
    
#         polylines_aff[i] = moved_polyline
    
#     polylines_aff0 = polylines_aff.copy()  
    
    
#%% registration - quad dwich ICP

polylines_quad = polylines_aff.copy()
degree = 2
init = 'identity'

reg = register.reg_polynom(niter=icp_niter, degree=degree, init=init, se=True, bidir=bidir, plot=False)

for i in range(1, nslice-1):
    
    prev_polyline = polylines_quad[i-1]
    next_polyline = polylines_aff[i+1]
    mov_polyline = polylines_aff[i]

    moved_polyline_prev = reg.compute(prev_polyline, mov_polyline)
    moved_polyline_next = reg.compute(next_polyline, mov_polyline)
    
    moved_pts_prev, _, _, _ = moved_polyline_prev
    moved_pts_next, _, _, _ = moved_polyline_next
    moved_pts = (moved_pts_prev + moved_pts_next) / 2
    moved_polyline = moved_pts, mov_polyline[1], mov_polyline[2], mov_polyline[3]
    polylines_quad[i] = moved_polyline

# for i in reversed(range(1, nslice-1)):
    
#     prev_polyline = polylines_aff[i-1]
#     next_polyline = polylines_quad[i+1]
#     mov_polyline = polylines_quad[i]

#     moved_polyline_prev = reg.compute(prev_polyline, mov_polyline)
#     moved_polyline_next = reg.compute(next_polyline, mov_polyline)
    
#     moved_pts_prev, _, _ = moved_polyline_prev
#     moved_pts_next, _, _ = moved_polyline_next
#     moved_pts = (moved_pts_prev + moved_pts_next) / 2
#     moved_polyline = moved_pts, mov_polyline[1], mov_polyline[2]
#     polylines_quad[i] = moved_polyline

#  - 

# for i in range(midslice+1,nslice-1):
    
#     prev_polyline = polylines_quad[i-1]
#     next_polyline = polylines_aff[i+1]
#     mov_polyline = polylines_aff[i]

#     moved_polyline_prev = reg.compute(prev_polyline, mov_polyline)
#     moved_polyline_next = reg.compute(next_polyline, mov_polyline)
    
#     moved_pts_prev, _, _, _ = moved_polyline_prev
#     moved_pts_next, _, _, _ = moved_polyline_next
#     moved_pts = (moved_pts_prev + moved_pts_next) / 2
#     moved_polyline = moved_pts, mov_polyline[1], mov_polyline[2], mov_polyline[3]
#     polylines_quad[i] = moved_polyline

# for i in reversed(range(1,midslice)):
    
#     prev_polyline = polylines_quad[i+1]
#     next_polyline = polylines_aff[i-1]
#     mov_polyline = polylines_aff[i]

#     moved_polyline_prev = reg.compute(prev_polyline, mov_polyline)
#     moved_polyline_next = reg.compute(next_polyline, mov_polyline)
    
#     moved_pts_prev, _, _, _ = moved_polyline_prev
#     moved_pts_next, _, _, _ = moved_polyline_next
#     moved_pts = (moved_pts_prev + moved_pts_next) / 2
#     moved_polyline = moved_pts, mov_polyline[1], mov_polyline[2], mov_polyline[3]
#     polylines_quad[i] = moved_polyline
   
 
#  method 4

# polylines_quad0 = polylines_aff.copy()

# for _ in range(1):
    
#     for i in range(1, nslice, 2):
        
#         mov_polyline = polylines_quad0[i]
        
#         prev_polyline = polylines_quad0[i-1]
#         moved_polyline_prev = reg.compute(prev_polyline, mov_polyline)
        
#         if i < nslice - 1:
#             next_polyline = polylines_quad0[i+1]
#             moved_polyline_next = reg.compute(next_polyline, mov_polyline)
#         else:
#             moved_polyline_next = mov_polyline
            
#         moved_pts_prev, _, _, _ = moved_polyline_prev
#         moved_pts_next, _, _, _ = moved_polyline_next
#         moved_pts = (moved_pts_prev + moved_pts_next) / 2
#         moved_polyline = moved_pts, mov_polyline[1], mov_polyline[2], mov_polyline[3]
#         polylines_quad[i] = moved_polyline
    
#     polylines_quad0 = polylines_quad.copy()
    
#     for i in range(0, nslice, 2):
        
#         mov_polyline = polylines_quad0[i]
        
#         if i > 0:
#             prev_polyline = polylines_quad0[i-1]
#             moved_polyline_prev = reg.compute(prev_polyline, mov_polyline)
#         else:
#             moved_polyline_prev = mov_polyline
            
#         next_polyline = polylines_quad0[i+1]
#         moved_polyline_next = reg.compute(next_polyline, mov_polyline)
        
#         moved_pts_prev, _, _, _ = moved_polyline_prev
#         moved_pts_next, _, _, _ = moved_polyline_next
#         moved_pts = (moved_pts_prev + moved_pts_next) / 2
#         moved_polyline = moved_pts, mov_polyline[1], mov_polyline[2], mov_polyline[3]
#         polylines_quad[i] = moved_polyline
        
#     polylines_quad0 = polylines_quad.copy()
  
    
#  method 5

polylines_quad0 = polylines_aff.copy()

for _ in range(3):
    
    for i in range(1, nslice-1):
        
        mov_polyline = polylines_quad0[i]
        
        ref_polyline = [polylines_quad0[i-1]]
        ref_polyline += [polylines_quad0[i+1]]
            
        ref_polyline = utils.concat_contours(ref_polyline)
        moved_polyline = reg.compute(ref_polyline, mov_polyline)
    
        polylines_quad[i] = moved_polyline
        
    polylines_quad0 = polylines_quad.copy()
 
    
#%% mesh reconstruction and save

z_coords = np.arange(len(polylines_raw))

# polylines_quad = polylines_quad.copy()[65:75]
# z_coords = z_coords[65:75]
# nslice = len(z_coords)

polylines_all = []
suffix = []
if save_raw: polylines_all.append(polylines_raw); suffix.append('raw')
if save_rig: polylines_all.append(polylines_rig); suffix.append('rig')
if save_aff: polylines_all.append(polylines_aff); suffix.append('aff')
if save_quad: polylines_all.append(polylines_quad); suffix.append('quad')
nreg = len(polylines_all)

opts_lists = []
for j in range(nreg):
    polylines =  polylines_all[j]
    opts_lists.append([utils.contours2opts(np.array(polyline[0]), np.array(polyline[1])) for polyline in polylines])

for i in range(nslice):
    for j in range(nreg):
        polylines = polylines_all[j]
        plt.subplot(1,nreg,j+1)
        utils.plot_contour(polylines[i], xlim=[-2,2], ylim=[-2,2])
        plt.title(suffix[j])
    plt.suptitle('slice ' + str(i), y=0.8)
    plt.tight_layout()
    plt.show()
    
for j in range(nreg):
    
    polylines = polylines_all[j]

    for l in range(nlabs):
        opts_list = []
        for i in range(nslice):
            polyline = polylines[i]
            polyline_l = utils.extract_polyline(polyline, polyline[3] == l + 1)
            opts = utils.contours2opts(np.array(polyline_l[0]), np.array(polyline_l[1]))
            opts_list.append(opts)
            
        pts, simps = utils.bridge_contours_2(opts_list, z_coords, greedy=False, thr_conn=thr_conn[l])
    
        pts[:,:2] = (pts[:,:2] * pts_amp + pts_mu) * voxdim[:2]
        pts[:,2] = pts[:,2] * voxdim[2]
        
        poly = utils.vtkpoly(pts, simps)
        out_file = os.path.join(out_dir, contours_name[l] + '_' + suffix[j] + '_5-3.obj')
        utils.write_vtkpoly(poly, out_file)
