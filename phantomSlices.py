wkdir = '/Users/alegouhy/tests/contours2mesh'
import os
os.chdir(wkdir)
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle

import contours2mesh as c2m
import transfo as transfo_ops
import energy
import utils
cols = utils.get_cols()

#%%

datadir = '/Users/alegouhy/data/polygon2mesh/p32-data'
names = ['mesh-4_sym_left']
# datadir = '/Users/alegouhy/data/polygon2mesh/bananas'
# names = ['banana_simple']
out_mesh = True

outdir = '/Users/alegouhy/tests/contours2mesh/output/test_3'
os.makedirs(outdir, exist_ok=True)
os.makedirs(os.path.join(outdir, 'polylines'), exist_ok=True)
os.makedirs(os.path.join(outdir, 'meshes'), exist_ok=True)

axis = 1
nslice = 100
npts = 100
thr_conn = 0.5

#%% load and slice mesh

io = c2m.io(datadir, names, npts=npts)
if out_mesh:
    mesher = c2m.bridge_contours(thr_conn=thr_conn, sealed=True)

polylines_gt = io.load2(axis=axis, nslice=nslice, plot=True, ext='ply')
if out_mesh:
    meshes_gt = mesher.compute(polylines_gt)
    io.save(meshes_gt, os.path.join(outdir, 'meshes'), suffix='gt')
filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_gt.pkl')
with open(filename, 'wb') as f:
    pickle.dump(polylines_gt, f)

#%% deform slices

ncpts = 30
sigmap = 0.05
bound_scal = 1.1
ne_corner = np.array([-1,-1])
so_corner = np.array([1,1])
 
trans_bounds=0.05
rot_bounds=np.pi/20
scalDir_bounds=np.pi/4
scal_bounds=1.05
trans_bounds_loc=0.1
rot_bounds_loc=np.pi/12
scalDir_bounds_loc=np.pi/4
scal_bounds_loc=1.1

sigma = 0.4
int_steps = 8

polya = transfo_ops.kernel_disp(sigma=sigma, int_steps=int_steps)

polylines_synth = []
for i in range(nslice):
    
    pts, simps, _, _ = polylines_gt[i]

    # global affine:
    _, theta_lin, theta_trans = transfo_ops.random_locAff(ndims=2, ncpts=1, liealg=False,
                                                          so_corner=[0,0], ne_corner=[0,0], bound_scal=0,
                                                          trans_bounds=trans_bounds, rot_bounds=rot_bounds, 
                                                          scalDir_bounds=scalDir_bounds, scal_bounds=scal_bounds)
    pts_moved = pts @ theta_lin[0,:,:].T + theta_trans[0,:]
    pts_moved = jnp.array(pts_moved)
    
    # polyaffine:
    cpts, theta_lin, theta_trans = transfo_ops.random_locAff(ndims=2, ncpts=ncpts,
                                                             so_corner=so_corner, ne_corner=ne_corner, bound_scal=bound_scal,
                                                             trans_bounds=trans_bounds_loc, rot_bounds=rot_bounds_loc, 
                                                             scalDir_bounds=scalDir_bounds_loc, scal_bounds=scal_bounds_loc)
    cpts = jnp.array(cpts)
    theta_lin = None # jnp.array(theta_lin)
    theta_trans = jnp.array(theta_trans)
    
    disp = polya.compute(pts_moved, cpts, theta_lin=theta_lin, theta_trans=theta_trans)
    pts_moved = pts_moved + disp
    
    polyline = pts_moved, simps, None, None
    polylines_synth.append(polyline)

    plt.plot(cpts[:,0], cpts[:,1], '.', color=[1,0,0])
    plt.plot(pts[:,0], pts[:,1], '.', color=[0,0,1])
    plt.plot(pts_moved[:,0], pts_moved[:,1], '.', color=[0,0,0.5])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

if out_mesh:
    meshes_synth = mesher.compute(polylines_synth)
    io.save(meshes_synth, os.path.join(outdir, 'meshes'), suffix='synth')
filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_synth.pkl')
with open(filename, 'wb') as f:
    pickle.dump(polylines_synth, f)

#%% registration

bidir = True

for niter in [1,5]:   
    for method in [1]: # range(1,6):
        
        transfo = 'rigid'
        print('method ' + str(method) + ', ' + transfo)
        init = 'centroid'
        reg = c2m.register_slices(method, transfo, init, bidir=bidir)
        polylines_rig = reg.compute(polylines_synth)
        if out_mesh:
            meshes_rig = mesher.compute(polylines_rig)
            io.save(meshes_rig, os.path.join(outdir, 'meshes'), suffix='rig_met-'+str(method)+'-'+str(niter))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_rig_met'+'-'+str(method)+'-'+str(niter)+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(polylines_rig, f)   
        
        transfo = 'affine'
        print('method ' + str(method) + ', ' + transfo)
        init = 'identity'
        reg = c2m.register_slices(method, transfo, init, bidir=bidir)
        polylines_aff = reg.compute(polylines_rig)
        if out_mesh:
            meshes_aff = mesher.compute(polylines_aff)
            io.save(meshes_aff, os.path.join(outdir, 'meshes'), suffix='aff_met-'+str(method)+'-'+str(niter))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_aff_met'+'-'+str(method)+'-'+str(niter)+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(polylines_aff, f)
        
        transfo = 'polynomial'
        degree = 2
        init = 'identity'
        print('method ' + str(method) + ', ' + transfo)
        reg = c2m.register_slices(method, transfo, init, degree=degree, bidir=bidir)
        polylines_quad = reg.compute(polylines_aff)
        if out_mesh:
            meshes_quad = mesher.compute(polylines_quad)
            io.save(meshes_quad, os.path.join(outdir, 'meshes'), suffix='quad_met-'+str(method)+'-'+str(niter))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_quad_met'+'-'+str(method)+'-'+str(niter)+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(polylines_quad, f)
        
        transfo = 'deformable'
        print('method ' + str(method) + ', ' + transfo)
        lr = 1e-2
        wreg = 1e-2
        int_steps = 16
        sigma = 1e-1
        icp_niter = 50
        fit_fun = energy.pointdist(agg='mean', l=2)
        # regul_fun = energy.alap(transfo='similarity', normtype='l2')
        regul_fun = energy.grad_disp(l=2)
        reg = c2m.register_slices(method, transfo, fit_fun=fit_fun, regul_fun=regul_fun, niter=niter,
                                  icp_niter=icp_niter, lr=lr, wreg=wreg, sigma=sigma, int_steps=int_steps) #, plot=False)
        polylines_defo = reg.compute(polylines_aff)
        if out_mesh:
            meshes_defo = mesher.compute(polylines_defo)
            io.save(meshes_defo, os.path.join(outdir, 'meshes'), suffix='defo_met-'+str(method)+'-'+str(niter))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_defo_met'+'-'+str(method)+'-'+str(niter)+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(polylines_defo, f)
        # for i, polyline in enumerate(polylines_defo):
        #     utils.plot_contour(polyline)
        #     plt.title(i); plt.show()


#%% 1-to-1 point distance

# suffixes = ['synth', 'rig', 'aff', 'quad', 'defo-1', 'defo-5']
suffixes = ['synth', 'rig-1', 'rig-5', 'aff-1', 'aff-5', 'quad-1', 'quad-5', 'defo-1', 'defo-5']
suffixes = ['synth', 'rig-1', 'aff-1', 'quad-1', 'defo-1', 'defo-5']
metric_name = 'recovery'
center = False
do_exp = False
nt = len(suffixes)
methods = [1,2,3,4,5]
nm = len(methods)

filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_gt.pkl')
with open(filename, 'rb') as f:
    polylines_gt = pickle.load(f)
pts_gt = [polyline[0] for polyline in polylines_gt]
mu_gt = np.mean([np.mean(pt, axis=0) for pt in pts_gt], axis=0)
if center: pts_gt = [pt - mu_gt for pt in pts_gt]

dists = np.zeros((nt, nm, nslice), np.float32)
for m in range(nm):
    for t in range(nt):
        
        suffix = suffixes[t]
        if suffix == 'synth':
            filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix+'.pkl')
        else:
            filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix[:-2]+'_met'+'-'+str(methods[m])+'-'+suffix[-1]+'.pkl')
        
        with open(filename, 'rb') as f:
            polylines = pickle.load(f)
            
        pts = [polyline[0] for polyline in polylines]
        mu = np.mean([np.mean(pt, axis=0) for pt in pts], axis=0) 
        if center: pts = [pt - mu for pt in pts]
        dist = [np.mean(np.sqrt(np.sum((pts[k] - pts_gt[k])**2, axis=1))) for k in range(nslice)]
        dists[t,m,:] = dist

recov = np.zeros((nt, nm, nslice), np.float32)
for t in range(nt):       
    recov[t,:,:] = (dists[0,:,:] - dists[t,:,:]) / dists[0,:,:]
# recov = np.zeros((nt, nm, nslice), np.float32)
# for m in range(nm):
#     for t in range(1, nt):  # Skip t=0 (synth itself)
#         recov[t,m,:] = (dists[0,m,:] - dists[t,m,:]) / dists[0,m,:]
        
if metric_name == 'recovery':
    metric = recov
    ylims = [-0.15,0.75] if not do_exp else [0, 2]
elif metric_name == 'error':
    metric = dists
    ylims = [0,0.1]
if do_exp:
    metric = np.exp(metric)

# plt.figure(dpi=300, figsize=(8,5))
# tt = 1
# for t in range(nt):
#     plt.subplot(2,nt//2+1,tt)
#     for m in range(nm):
#         plt.plot(metric[t,m,:])
#     plt.title(suffixes[t])    
#     plt.tight_layout()
#     tt = tt+2 if t==2 else tt+1
# plt.suptitle(metric_name)
# plt.show()

plt.figure(dpi=300, figsize=(8,5))
for m in range(nm):
    plt.subplot(2,nm // 2 + 1,m+1)
    for s in range(nslice):
        plt.plot(metric[:,m,s], color=[0.9]*3)
    for t in range(nt):
        utils.boxplot(metric[t,m,:], t, cols[t,:])
    # plt.plot([-0.5, nt+0.5],[0,0])
    plt.title('method '+str(methods[m]))
    plt.xticks(np.arange(nt), suffixes, rotation=45, ha='right')
    plt.grid(axis='y')
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
    plt.ylim(ylims)
plt.suptitle(metric_name, y=1.02)
plt.show()


plt.figure(dpi=300, figsize=(4,4))
for s in range(nslice):
    plt.plot(metric[-1,:,s], color=[0.9]*3)
for m in range(nm):
    utils.boxplot(metric[-1,m,:], m, cols[m,:])
    plt.plot([-0.5, nm-0.5],[0,0])
    plt.xticks(np.arange(nm), np.arange(nm))
    plt.ylim(ylims)
    plt.grid(axis='y')
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
plt.suptitle(metric_name, y=1.02)
plt.show()

# m = 5
# plt.plot(dists[0,m,:], color=[0,0,1])
# plt.plot(dists[5,m,:], color=[1,0,0])
# plt.show()



#%% global desc.

from scipy.spatial.distance import jensenshannon
import pyvista as pv

suffixes = ['synth', 'rig-1', 'aff-1', 'defo-1', 'defo-5']
transfo_names = ['synth', 'rig', 'aff', 'defo', 'defo-5']
center = True
nt = len(suffixes)
methods = [0,2,3,4,5]
method_names = ['A', 'B', 'C', 'D']
nm = len(methods)

n_sample = 1000
def desc_1(pts, ind_pts):
    pts_sample_1 = pts[ind_pts[0,:], :]
    pts_sample_2 = pts[ind_pts[1,:], :]
    dists = np.sum((pts_sample_1 - pts_sample_2) ** 2, axis=1)
    return np.sqrt(dists)

k = 100
def desc_3(vtkpoly):

    mesh = pv.wrap(vtkpoly)
    mesh = mesh.compute_normals(point_normals=True, 
                                cell_normals=False,
                                split_vertices=False,
                                auto_orient_normals=True)
    curv_mean = mesh.curvature('mean')
    curv_gauss = mesh.curvature('gaussian')
    normals = mesh.point_data['Normals']

    return curv_mean, curv_gauss, normals

def histo(y, bin_edges, q=1):
    y_clipped = np.clip(y, bin_edges[0], bin_edges[-1])
    counts, bins = np.histogram(y_clipped, bins=bin_edges)
    hist = counts.astype(np.float32) / np.sum(counts)
    return hist

def bins_edges(y, nbins, q=0.95):
    y_clean = remove_outliers(y, q)
    _, bin_edges = np.histogram(y_clean, bins=nbins)
    return bin_edges

def spherical_hist(normals, k):
    theta = np.arctan2(normals[:, 1], normals[:, 0])
    phi = np.arccos(np.clip(normals[:, 2], -1, 1))
    hist, _, _ = np.histogram2d(theta, phi, 
                                bins=[k, k//2],
                                range=[[-np.pi, np.pi], [0, np.pi]])
    hist = hist.flatten()
    hist = hist / hist.sum()
    return hist

def remove_outliers(data, q=0.95):
    lower = np.quantile(data, 1 - q)
    upper = np.quantile(data, q)
    return data[(data >= lower) & (data <= upper)]

polylines_file = os.path.join(outdir, 'polylines', names[0] + '_polylines_gt.pkl')
with open(polylines_file, 'rb') as f:
    polylines_gt = pickle.load(f)
opts_gt = np.concatenate([polyline[0] for polyline in polylines_gt], axis=0)
nopts = opts_gt.shape[0]
ind_pts = np.random.randint(0,nopts,(2,n_sample))

mesh_file_gt = os.path.join(outdir, 'meshes', names[0] + '_gt.obj')
mesh_gt = utils.read_vtkpoly(mesh_file_gt)
pts_gt, simps_gt = utils.vtkpoly2mesh(mesh_gt)

desc_1_gt = desc_1(opts_gt, ind_pts)
curv_mean_gt, curv_gauss_gt, normals_gt = desc_3(mesh_gt)

hist_normals_gt = spherical_hist(normals_gt, k // 5)
bins_curv_mean_gt = bins_edges(curv_mean_gt, k)
bins_curv_gauss_gt = bins_edges(curv_gauss_gt, k) 
hist_curv_mean_gt = histo(curv_mean_gt, bins_curv_mean_gt)
hist_curv_gauss_gt = histo(curv_gauss_gt, bins_curv_gauss_gt)

dist_p2p = np.zeros((nt, nm, nopts), np.float32)
dist_curv_gauss = np.zeros((nt, nm), np.float32)
dist_curv_mean = np.zeros((nt, nm), np.float32)
dist_normals = np.zeros((nt, nm), np.float32)

pts_list = []
simps_list = []
for m in range(nm):
    for t in range(nt):
        suffix = suffixes[t]

        if suffix == 'synth':
            polylines_file = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix+'.pkl')
            mesh_file = os.path.join(outdir, 'meshes', names[0] + '_'+suffix+'.obj')
        else:
            polylines_file = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix[:-2]+'_met'+'-'+str(methods[m])+'-'+suffix[-1]+'.pkl')
            mesh_file = os.path.join(outdir, 'meshes', names[0] + '_'+suffix[:-2]+'_met'+'-'+str(methods[m])+'-'+suffix[-1]+'.obj')
        with open(polylines_file, 'rb') as f:
            polylines = pickle.load(f) 
            
        opts = np.concatenate([polyline[0] for polyline in polylines], axis=0)
        dist_p2p[t, m, :] = np.sqrt(np.sum((opts - opts_gt) ** 2, axis=1))

        mesh = utils.read_vtkpoly(mesh_file)
        pts, simps = utils.vtkpoly2mesh(mesh)
        
        curv_mean, curv_gauss, normals = desc_3(mesh)
        hist_normals = spherical_hist(normals, k // 5)
        hist_curv_mean = histo(curv_mean, bins_curv_mean_gt)
        hist_curv_gauss = histo(curv_gauss, bins_curv_gauss_gt)
        
        dist_normals[t, m] = jensenshannon(hist_normals_gt, hist_normals)
        dist_curv_mean[t, m] = jensenshannon(hist_curv_mean_gt, hist_curv_mean)
        dist_curv_gauss[t, m] = jensenshannon(hist_curv_gauss_gt, hist_curv_gauss)

metric_names = ['p2p', 'H', 'K', 'N']

ylims = [[0,0.1], [0,0.1], [0,0.005], [0,0.005], [0,0.01], [0,0.01], [0,0.01], [0,0.006], [0,0.45]]

ns = dist_p2p.shape[2]


plt.figure(dpi=300, figsize=(9,2.5))
plt.subplot(1, nm+1 , 1)
utils.boxplot(dist_p2p[0,m,:], 1, cols[0,:])
plt.title('synth')
# plt.xticks(np.arange(nt), suffixes)
plt.grid(axis='y')
plt.gca().set_axisbelow(False)
plt.tight_layout()
plt.ylim([0,0.1])
plt.xlim([0.5, 4.5])

for m in range(nm):
    plt.subplot(1, nm+1 , m+2)
    for t in range(1,nt):
        utils.boxplot(dist_p2p[t,m,:], t, cols[t,:])
    plt.title('method '+str(methods[m]))
    # plt.xticks(np.arange(nt), suffixes)
    plt.grid(axis='y')
    plt.gca().set_axisbelow(False)
    plt.ylim([0,0.1])
    plt.xlim([0.5, 4.5])
plt.tight_layout()
plt.suptitle(metric_names[0], y=1.02)
plt.savefig(os.path.join(outdir, 'results_p2p.svg'))
plt.show()


metrics = [dist_curv_mean, dist_curv_gauss, dist_normals]
nmet = len(metrics)
plt.figure(dpi=300, figsize=(9,2.5))

plt.subplot(1, nm+1, 1)
for i in range(len(metrics)):
    metric = metrics[i]
    pos = i
    plt.bar(pos, metric[0,m], color=cols[0,:], width=0.17)
plt.title('synth')
plt.xticks(np.arange(nmet), metric_names[1:])
plt.grid(axis='y')
plt.gca().set_axisbelow(False)
plt.xlim([-1,3])
plt.tight_layout()
plt.xlim([-0.5,2.5])
plt.ylim([0, 0.45])

for m in range(nm):
    plt.subplot(1, nm+1, m+2)
    for t in range(1,nt):
        for i in range(len(metrics)):
            metric = metrics[i]
            pos = i + (t-2.5) * 0.2
            plt.bar(pos, metric[t,m], color=cols[t,:], width=0.17)
    plt.title('method '+str(methods[m]))
    plt.xticks(np.arange(nmet), metric_names[1:])
    plt.grid(axis='y')
    plt.gca().set_axisbelow(False)
    plt.tight_layout()
    plt.xlim([-0.5,2.5])
    plt.ylim([0, 0.45])
plt.savefig(os.path.join(outdir, 'results_desc.svg'))
plt.show()

metrics = [dist_curv_mean, dist_curv_gauss, dist_normals]
nmet = len(metrics)
plt.figure(dpi=300, figsize=(8,3))

plt.show()


#%% global desc.

from scipy.spatial.distance import jensenshannon
import pyvista as pv

suffixes = ['synth', 'rig-1', 'aff-1', 'quad-1', 'defo-1', 'defo-5']
metric_name = 'recovery'
center = True
do_exp = False
nt = len(suffixes)
methods = [1,2,3,4,5]
nm = len(methods)

n_sample = 1000
def desc_1(pts, ind_pts):
    pts_sample_1 = pts[ind_pts[0,:], :]
    pts_sample_2 = pts[ind_pts[1,:], :]
    dists = np.sum((pts_sample_1 - pts_sample_2) ** 2, axis=1)
    return np.sqrt(dists)

def desc_2(pts, ind_pts):
    pts_sample = pts[ind_pts[0,:], :]
    pts_bar = pts_sample - pts_sample.mean(axis=0)
    gram = pts_bar @ pts_bar.T
    eigval = np.linalg.eigvalsh(gram)
    eigval = np.sort(eigval)[::-1]
    return eigval[:2]

k = 100
def desc_3(vtkpoly):

    mesh = pv.wrap(vtkpoly)
    mesh = mesh.compute_normals(point_normals=True, 
                                cell_normals=False,
                                split_vertices=False,
                                auto_orient_normals=True)
    curv_mean = mesh.curvature('mean')
    curv_gauss = mesh.curvature('gaussian')
    normals = mesh.point_data['Normals']

    return curv_mean, curv_gauss, normals

def spherical_hist(normals, k):
    theta = np.arctan2(normals[:, 1], normals[:, 0])
    phi = np.arccos(np.clip(normals[:, 2], -1, 1))
    hist, _, _ = np.histogram2d(theta, phi, 
                                bins=[k, k//2],
                                range=[[-np.pi, np.pi], [0, np.pi]])
    hist = hist.flatten()
    hist = hist / hist.sum()
    return hist

def remove_outliers(data, percentile=95):
    lower = np.percentile(data, 100 - percentile)
    upper = np.percentile(data, percentile)
    return data[(data >= lower) & (data <= upper)]

polylines_file = os.path.join(outdir, 'polylines', names[0] + '_polylines_gt.pkl')
with open(polylines_file, 'rb') as f:
    polylines_gt = pickle.load(f)
opts_gt = np.concatenate([polyline[0] for polyline in polylines_gt], axis=0)
nopts = opts_gt.shape[0]
ind_pts = np.random.randint(0,nopts,(2,n_sample))

mesh_file_gt = os.path.join(outdir, 'meshes', names[0] + '_gt.obj')
mesh_gt = utils.read_vtkpoly(mesh_file_gt)
pts_gt, simps_gt = utils.vtkpoly2mesh(mesh_gt)

desc_1_gt = desc_1(opts_gt, ind_pts)
# desc_2_gt = desc_2(opts_gt, ind_pts)
curv_mean_gt, curv_gauss_gt, normals_gt = desc_3(mesh_gt)
hist_normals_gt = spherical_hist(normals_gt, k // 5)
hist_curv_mean_gt, bins_curv_mean = np.histogram(remove_outliers(curv_mean_gt), bins=k)
hist_curv_mean_gt = hist_curv_mean_gt.astype(np.float32) / np.sum(hist_curv_mean_gt)
hist_curv_gauss_gt, bins_curv_gauss = np.histogram(remove_outliers(curv_gauss_gt), bins=k)
hist_curv_gauss_gt = hist_curv_gauss_gt.astype(np.float32) / np.sum(hist_curv_gauss_gt)
hist_normals_gt_1, bins_normals_1 = np.histogram(remove_outliers(normals_gt[:,0]), bins=k)
hist_normals_gt_1 = hist_normals_gt_1.astype(np.float32) / np.sum(hist_normals_gt_1)
hist_normals_gt_2, bins_normals_2 = np.histogram(remove_outliers(normals_gt[:,1]), bins=k)
hist_normals_gt_2 = hist_normals_gt_2.astype(np.float32) / np.sum(hist_normals_gt_2)
hist_normals_gt_3, bins_normals_3 = np.histogram(remove_outliers(normals_gt[:,2]), bins=k)
hist_normals_gt_3 = hist_normals_gt_3.astype(np.float32) / np.sum(hist_normals_gt_3)
plt.subplot(2,3,1); plt.ylim(0,0.075)
plt.plot(bins_curv_mean[1:], hist_curv_mean_gt)
plt.title('mean curv')
plt.subplot(2,3,2); plt.ylim(0,0.075)
plt.plot(bins_curv_gauss[1:], hist_curv_gauss_gt)
plt.title('gauss curv')
plt.subplot(2,3,4); plt.ylim(0,0.075)
plt.plot(bins_normals_1[1:], hist_normals_gt_1)
plt.title('normal_1')
plt.subplot(2,3,5); plt.ylim(0,0.075)
plt.plot(bins_normals_2[1:], hist_normals_gt_2)
plt.title('normal_2')
plt.subplot(2,3,6); plt.ylim(0,0.075)
plt.plot(bins_normals_3[1:], hist_normals_gt_3)
plt.title('normal_3')
plt.suptitle('GT'); plt.tight_layout(); plt.show()

dist_0 = np.zeros((nt, nm, nopts), np.float32)
dist_1 = np.zeros((nt, nm, n_sample), np.float32)
dist_2 = np.zeros((nt, nm, 2), np.float32)
dist_3 = np.zeros((nt, nm, k), np.float32)
dist_4 = np.zeros((nt, nm, k), np.float32)
dist_5 = np.zeros((nt, nm, k), np.float32)
dist_6 = np.zeros((nt, nm, k), np.float32)
dist_7 = np.zeros((nt, nm, k), np.float32)
dist_8 = np.zeros((nt, nm, int((k/5)*((k/5)//2))), np.float32)
dist_9 = np.zeros((nt, nm, 1), np.float32)

pts_list = []
simps_list = []
for m in range(nm):
    for t in range(nt):
        suffix = suffixes[t]

        if suffix == 'synth':
            polylines_file = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix+'.pkl')
            mesh_file = os.path.join(outdir, 'meshes', names[0] + '_'+suffix+'.obj')
        else:
            polylines_file = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix[:-2]+'_met'+'-'+str(methods[m])+'-'+suffix[-1]+'.pkl')
            mesh_file = os.path.join(outdir, 'meshes', names[0] + '_'+suffix[:-2]+'_met'+'-'+str(methods[m])+'-'+suffix[-1]+'.obj')
        with open(polylines_file, 'rb') as f:
            polylines = pickle.load(f) 
        opts = np.concatenate([polyline[0] for polyline in polylines], axis=0)
        
        mesh = utils.read_vtkpoly(mesh_file)
        pts, simps = utils.vtkpoly2mesh(mesh)
        
        dist_0[t, m, :] = np.sqrt(np.sum((opts - opts_gt) ** 2, axis=1))
        dist_1[t, m, :] = np.abs(desc_1_gt - desc_1(opts, ind_pts))
        
        curv_mean, curv_gauss, normals = desc_3(mesh)
        hist_normals = spherical_hist(normals, k // 5)
        hist_curv_mean, _ = np.histogram(remove_outliers(curv_mean), bins=bins_curv_mean)
        hist_curv_mean = hist_curv_mean.astype(np.float32) / np.sum(hist_curv_mean)
        hist_curv_gauss, _ = np.histogram(remove_outliers(curv_gauss), bins=bins_curv_gauss)
        hist_curv_gauss = hist_curv_gauss.astype(np.float32) / np.sum(hist_curv_gauss)
        hist_normals_1, _ = np.histogram(remove_outliers(normals[:,0]), bins=bins_normals_1)
        hist_normals_1 = hist_normals_1.astype(np.float32) / np.sum(hist_normals_1)
        hist_normals_2, _ = np.histogram(remove_outliers(normals[:,1]), bins=bins_normals_2)
        hist_normals_2 = hist_normals_2.astype(np.float32) / np.sum(hist_normals_2)
        hist_normals_3, _ = np.histogram(remove_outliers(normals[:,2]), bins=bins_normals_3)
        hist_normals_3 = hist_normals_3.astype(np.float32) / np.sum(hist_normals_3)
        dist_3[t, m, :] = np.abs(hist_curv_mean - hist_curv_mean_gt)
        dist_4[t, m, :] = np.abs(hist_curv_gauss - hist_curv_gauss_gt)
        dist_5[t, m, :] = np.abs(hist_normals_1 - hist_normals_gt_1)
        dist_6[t, m, :] = np.abs(hist_normals_2 - hist_normals_gt_2)
        dist_7[t, m, :] = np.abs(hist_normals_3 - hist_normals_gt_3)
        dist_8[t, m, :] = np.abs(hist_normals - hist_normals_gt)
        dist_9[t, m, :] = jensenshannon(hist_normals_gt, hist_normals)
        
        plt.subplot(2,3,1); plt.ylim(0,0.075)
        plt.plot(bins_curv_mean[1:], hist_curv_mean_gt)
        plt.stairs(hist_curv_mean, bins_curv_mean, fill=True)
        plt.title('mean curv')
        plt.subplot(2,3,2); plt.ylim(0,0.075)
        plt.plot(bins_curv_gauss[1:], hist_curv_gauss_gt)
        plt.stairs(hist_curv_gauss, bins_curv_gauss, fill=True)
        plt.title('gauss curv')
        plt.subplot(2,3,4); plt.ylim(0,0.075)
        plt.plot(bins_normals_1[1:], hist_normals_gt_1)
        plt.stairs(hist_normals_1 ,bins_normals_1, fill=True)
        plt.title('normal_1')
        plt.subplot(2,3,5); plt.ylim(0,0.075)
        plt.plot(bins_normals_2[1:], hist_normals_gt_2)
        plt.stairs(hist_normals_2, bins_normals_2, fill=True)
        plt.title('normal_2')
        plt.subplot(2,3,6); plt.ylim(0,0.075)
        plt.plot(bins_normals_3[1:], hist_normals_gt_3)
        plt.stairs(hist_normals_3, bins_normals_3, fill=True)
        plt.title('normal_3')
        plt.subplot(2,3,3); plt.ylim(0,0.075)
        plt.stairs(hist_normals)
        plt.plot(hist_normals_gt)
        plt.title('normal')
        plt.suptitle(suffix); plt.tight_layout(); plt.show()

metric_names = ['point-to-point dists', 'pairwise dists', 'mean curv', 'gauss curv', 'normals 1', 'normals 2', 'normals 3', 'normals', 'jensenshannon']
metrics = [dist_0, dist_1, dist_3, dist_4, dist_5, dist_6, dist_7, dist_8, dist_9]
ylims = [[0,0.1], [0,0.1], [0,0.005], [0,0.005], [0,0.01], [0,0.01], [0,0.01], [0,0.006], [0,0.45]]

recovery = False
if recovery: ylims = [[-0.5,1]]

for i in range(len(metrics)):
    
    metric = metrics[i].copy()
    if recovery:
        for t in range(nt):       
            metric[t,:,:] = (metrics[i][0,:,:] - metrics[i][t,:,:]) / (metrics[i][0,:,:] + 1e-9)
    ns = metric.shape[2]
    
    plt.figure(dpi=300, figsize=(8,5))
    for m in range(nm):
        plt.subplot(2,nm // 2 + 1,m+1)
        # for s in range(ns):
        #     plt.plot(metric[:,m,s], color=[0.9]*3)
        for t in range(nt):
            utils.boxplot(metric[t,m,:], t, cols[t,:])
        # plt.plot([-0.5, nt+0.5],[0,0])
        plt.title('method '+str(methods[m]))
        plt.xticks(np.arange(nt), suffixes, rotation=45, ha='right')
        plt.grid(axis='y')
        plt.gca().set_axisbelow(False)
        plt.tight_layout()
        plt.ylim(ylims[i])
    plt.suptitle(metric_names[i], y=1.02)
    plt.show()


