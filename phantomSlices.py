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

outdir = '/Users/alegouhy/tests/contours2mesh/output/test_1'

axis = 1
nslice = 100
npts = 100 # 300
thr_conn = 0.5

#%% load and slice mesh

io = c2m.io(datadir, names, npts=npts)
if out_mesh:
    mesher = c2m.bridge_contours(thr_conn=thr_conn, sealed=True)

polylines_gt = io.load2(axis=axis, nslice=nslice, plot=True, ext='ply')
if out_mesh:
    meshes_gt = mesher.compute(polylines_gt)
    io.save(meshes_gt, outdir, suffix='gt')
filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_gt.pkl')
with open(filename, 'wb') as f:
    pickle.dump(polylines_gt, f)

#%% deform slices

ncpts = 30
sigmap = 0.05
bound_scal = 1.1
ne_corner = np.array([-1,-1])
so_corner = np.array([1,1])
 
trans_bounds=0.1
rot_bounds=np.pi/8
scalDir_bounds=np.pi/12
scal_bounds=1.1
trans_bounds_loc=0.3
rot_bounds_loc=0 # np.pi/6
scalDir_bounds_loc=0 # np.pi/8
scal_bounds_loc=1 # 1.5

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
    pts_moved = pts #@ theta_lin[0,:,:].T + theta_trans[0,:]
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
    io.save(meshes_synth, outdir, suffix='synth')
filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_synth.pkl')
with open(filename, 'wb') as f:
    pickle.dump(polylines_synth, f)

#%% registration

bidir = True

for niter in [5]:
    
    for method in range(1,6):
        
        transfo = 'rigid'
        print('method ' + str(method) + ', ' + transfo)
        init = 'centroid'
        reg = c2m.register_slices(method, transfo, init, bidir=bidir)
        polylines_rig = reg.compute(polylines_synth)
        if out_mesh:
            meshes_rig = mesher.compute(polylines_rig)
            io.save(meshes_rig, outdir, suffix='rig_met-'+str(method))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_rig_met'+'-'+str(method)+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(polylines_rig, f)   
        
        transfo = 'affine'
        print('method ' + str(method) + ', ' + transfo)
        init = 'identity'
        reg = c2m.register_slices(method, transfo, init, bidir=bidir)
        polylines_aff = reg.compute(polylines_rig)
        if out_mesh:
            meshes_aff = mesher.compute(polylines_aff)
            io.save(meshes_aff, outdir, suffix='aff_met-'+str(method))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_aff_met'+'-'+str(method)+'.pkl')
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
            io.save(meshes_quad, outdir, suffix='quad_met-'+str(method))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_quad_met'+'-'+str(method)+'.pkl')
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
                                  icp_niter=icp_niter, lr=lr, wreg=wreg, sigma=sigma, int_steps=int_steps, plot=False)
        polylines_defo = reg.compute(polylines_aff)
        if out_mesh:
            meshes_defo = mesher.compute(polylines_defo)
            io.save(meshes_defo, outdir, suffix='defo_met-'+str(method)+'-'+str(niter))
        filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_defo_met'+'-'+str(method)+'-'+str(niter)+'.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(polylines_defo, f)
        # for i, polyline in enumerate(polylines_defo):
        #     utils.plot_contour(polyline)
        #     plt.title(i); plt.show()


#%%

suffixes = ['synth', 'rig', 'aff', 'quad', 'defo-1']
metric_name = 'recovery'
center = True
do_exp = False
nt = len(suffixes)
nm = 6

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
        elif suffix in ('rig', 'aff', 'quad'):
            filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_'+suffix+'_met'+'-'+str(m)+'.pkl')
        elif 'defo' in suffix:
            filename = os.path.join(outdir, 'polylines', names[0] + '_polylines_defo_met'+'-'+str(m)+'-'+suffix[-1]+'.pkl')
        
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
    ylims = [0,0.2]
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
    plt.subplot(2,nm // 2,m+1)
    for s in range(nslice):
        plt.plot(metric[:,m,s], color=[0.9]*3)
    for t in range(nt):
        utils.boxplot(metric[t,m,:], t, cols[t,:])
    # plt.plot([-0.5, nt+0.5],[0,0])
    plt.title('method '+str(m))
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


#%%

a_file = 'banana_simple_gt'
b_file = 'banana_simple_defo_met-4-5'

a_poly = utils.read_vtkpoly(os.path.join(outdir, a_file + '.obj'))
b_poly = utils.read_vtkpoly(os.path.join(outdir, b_file + '.obj'))
a_pts, a_simps = utils.vtkpoly2mesh(a_poly)
b_pts, b_simps = utils.vtkpoly2mesh(b_poly)

b_poly_a = utils.vtkpoly(b_pts, a_simps)
utils.write_vtkpoly(b_poly_a, os.path.join(outdir, b_file + '_ogsimps.obj'))

#%%

axis = 0

poly = utils.read_vtkpoly('/Users/alegouhy/data/polygon2mesh/p32-data/mesh-4_sym.ply')
pts, simps = utils.vtkpoly2mesh(poly)
mu = np.mean(pts, axis=0)
pos = mu[axis]
poly_1, poly_2 = utils.split_vtkpoly2(poly, pos, 0)
# utils.render_vtkpoly([poly_1])

utils.write_vtkpoly(poly_1, '/Users/alegouhy/data/polygon2mesh/p32-data/mesh-4_sym_right.ply')
utils.write_vtkpoly(poly_2, '/Users/alegouhy/data/polygon2mesh/p32-data/mesh-4_sym_left.ply')
