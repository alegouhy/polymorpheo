testdir = '/Users/alegouhy/tests/ferret_atlas_reg'
import os
os.chdir(testdir)
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import matplotlib.pyplot as plt
import numpy as np
import time

import utils
import energy
import contours2mesh as c2m
import micro2mri as m2M
import register

#%%

spacing_micro = np.array([0.1, 0.1, 1.25])
npts_micro = 100
npts_min_micro = 5

reg_met = 4

datadir_micro = '/Users/alegouhy/dev/polygons_to_mesh/data'
name_micro = 'lh_registered_contours'
micro_io = c2m.io(datadir_micro, [name_micro], spacing_micro, npts_micro, npts_min_micro)
polylines_raw = micro_io.load(plot=False)
slice_pos = np.arange(len(polylines_raw)) * spacing_micro[2]

datadir_mri = '/Users/alegouhy/data/polygon2mesh/p32-data'
name_mri = 'mesh-4_sym_left'
mesh_mri = utils.read_vtkpoly(os.path.join(datadir_mri, name_mri + '.ply'))
pts_mri = np.array(mesh_mri.GetPoints().GetData())
simps_mri = np.array(mesh_mri.GetPolys().GetData()).reshape((-1,4))[:,1:]

pts_micro_raw, simps_micro_raw = utils.polylines_2d_3d(polylines_raw, 2, slice_pos)
pts_micro_raw[:,:2] = pts_micro_raw[:,:2] * micro_io.pts_amp + micro_io.pts_mu

xlim = [micro_io.pts_mu[0] - micro_io.pts_amp / 4, micro_io.pts_mu[0] + micro_io.pts_amp / 4]
ylim = [micro_io.pts_mu[1] - micro_io.pts_amp / 4, micro_io.pts_mu[1] + micro_io.pts_amp / 4]

#%% micro serial slice registration

# rigid
reg = c2m.register_slices(reg_met, transfo='rigid', init='centroid')
polylines_rig = reg.compute(polylines_raw)

# affine
reg = c2m.register_slices(reg_met, transfo='affine')
polylines_aff = reg.compute(polylines_rig)

# deformable
t = time.time()
fit_fun = energy.point2point(agg='mean')
regul_fun = energy.grad_disp(l=2)
reg = c2m.register_slices(reg_met, transfo='deformable', fit_fun=fit_fun, regul_fun=regul_fun, niter=1,
                          icp_niter=50, lr=1e-2, wreg=5e-1, sigma=1e-1, int_steps=16, plot=False)
polylines_defo = reg.compute(polylines_aff)
print(time.time()-t)
pts_micro, simps_micro = utils.polylines_2d_3d(polylines_defo, 2, slice_pos)
pts_micro[:,:2] = pts_micro[:,:2] * micro_io.pts_amp + micro_io.pts_mu

fig = utils.plot_obj(pts_micro_raw, simps_micro, pts_col=(0,0,1), face_col=(0.5,0.5,1))
fig = utils.plot_obj(pts_micro+np.array([100,0,0]), simps_micro, pts_col=(1,0,0), face_col=(1,0.5,0.5), fig=fig)
fig.show()


#%% micro -> mri

# init 3D, mri -> micro
t = time.time()
pts_micro_init, _, _ = register.init_affcube(pts_mri, pts_micro)
print('cube init  -  dist:',utils.chamfer(pts_micro_init, pts_mri),',  time:', time.time()-t)
fig = utils.plot_obj(pts_mri, simps_mri)
fig = utils.plot_obj(pts_micro_init, simps_micro, pts_col=(1,0,0), face_col=(1,0.5,0.5), fig=fig)
fig.show()
mesh_mri = pts_mri, None, None, None
mesh_micro_init = pts_micro_init, simps_micro, None, None

# affine 3D, mri -> micro
reg_aff = register.reg_linear(niter=50, transfo='affine')
t = time.time()
_, mesh_micro_aff = reg_aff.compute(mesh_mri, mesh_micro_init)
pts_micro_aff = mesh_micro_aff[0]
print('affine  -  dist:',utils.chamfer(pts_micro_aff, pts_mri),',  time:', time.time()-t)
fig = utils.plot_obj(pts_mri, simps_mri)
fig = utils.plot_obj(pts_micro_aff+np.array([0,0,0]), simps_micro, pts_col=(1,0,0), face_col=(1, 0.5,0.5), fig=fig)
fig.show()

# deformable
mesh_micro_defo = mesh_micro_aff

lr = [1e-2, 1e-2, 1e-3]
wreg = [5e-1, 3e-1, 1e-1]
sigma = [5e-1, 1e-1, 5e-2]
cpts_ratio = [0.05, 0.1, 0.2]

for i in range(3):
    # fit_fun = energy.point2point(agg='mean', est='welsch', sigma=0.01, bidir=True)
    fit_fun = energy.point2point(agg='mean', alpha=-2, scale=0.01, bidir=True)
    # regul_fun = energy.grad_disp(l=2)
    regul_fun = energy.alap(transfo='similarity', l=2)
    regul_fun.set_neighs(mesh_micro_defo[1], mesh_micro_defo[0].shape[0])

    reg_defo = register.reg_deformable(niter=50, fit_fun=fit_fun, regul_fun=regul_fun,
                                       lr=lr[i], wreg=wreg[i], sigma=sigma[i], int_steps=8, rk=2, cpts_ratio=cpts_ratio[i])
    t = time.time()
    _, mesh_micro_defo, loss = reg_defo.compute(mesh_mri, mesh_micro_defo)
    plt.plot(loss)
    plt.show()
    pts_micro_defo = mesh_micro_defo[0]
    print('deformable  -  dist:',utils.chamfer(pts_micro_defo, pts_mri),',  time:', time.time()-t)
    fig = utils.plot_obj(pts_mri, simps_mri)
    fig = utils.plot_obj(pts_micro_defo+np.array([0,0,0]), simps_micro, pts_col=(1,0,0), face_col=(1,0.5,0.5), fig=fig)
    fig.show()


#%% micro <- mri

# init 3D, mri -> micro
pts_mri_init, _, _ = register.init_affcube(pts_micro,pts_mri)
print(utils.chamfer(pts_mri_init, pts_micro))
fig = utils.plot_obj(pts_mri_init, simps_mri)
fig = utils.plot_obj(pts_micro, simps_micro, pts_col=(1,0,0), face_col=(1,0.5,0.5), fig=fig)
fig.show()
mesh_mri_init = pts_mri_init, None, None, None
mesh_micro = pts_micro, None, None, None

# affine 3D, mri -> micro
reg_aff = register.reg_linear(niter=50, transfo='affine')
_, mesh_mri_aff = reg_aff.compute(mesh_micro, mesh_mri_init)
pts_mri_aff = mesh_mri_aff[0]
print(utils.chamfer(pts_mri_aff, pts_micro))
fig = utils.plot_obj(pts_mri_aff, simps_mri)
fig = utils.plot_obj(pts_micro+np.array([0,0,0]), simps_micro, pts_col=(1,0,0), face_col=(1, 0.5,0.5), fig=fig)
fig.show()


#%%






opts = utils.contours2opts(pts, simps)
polyline = utils.opts_to_contour(opts, npts=npts_micro)
polylines_mri_aff.append(polyline)

utils.plot_contour(polyline, xlim=xlim, ylim=ylim)
plt.show()


bounds = np.reshape(poly_mri_aff.GetBounds(),(3,2))
mu = np.array(poly_mri_aff.GetCenter())

# self.pts_amp = np.max(np.diff(np.delete(bounds, axis, axis=0))) / 2
# self.pts_mu = np.delete(mu, axis)
# pos_sli = np.linspace(bounds[axis,0], bounds[axis,1], nslice + 2)[1:-1]
# self.spacing = np.ones(3)
# self.spacing[axis] = pos_sli[1] - pos_sli[0]
# self.permut = list(range(axis)) + [2] + list(range(axis, 2))

# polylines = []
# for i in range(nslice):

#    poly_sli = utils.slice_vtkpoly(poly, pos_sli[i], axis=axis)
#    pts = np.array(poly_sli.GetPoints().GetData())
#    pts = np.delete(pts, axis, axis=1)
#    pts = (pts - self.pts_mu) / self.pts_amp
#    nsimps = poly_sli.GetNumberOfLines()
#    simps = np.array(poly_sli.GetLines().GetData()).reshape((nsimps,-1))
#    simps = np.array(simps, dtype=np.int32).reshape((nsimps,-1))[:,1:]

#    opts = utils.contours2opts(pts, simps)
#    polyline = utils.opts_to_contour(opts, npts=self.npts)
#    polylines.append(polyline)

#    if plot:
#        utils.plot_contour(polyline)
#        plt.title(str(i))
#        plt.show()

# return polylines




# # deformable
# pts_norm, mu, amp = utils.normalise_pts([pts_micro_aff, pts_mri])
# pts_micro_aff_norm, pts_mri_norm = pts_norm
# mesh_micro_aff_norm = pts_micro_aff_norm, simps_micro, None, None
# mesh_mri_norm = pts_mri_norm, simps_mri, None, None

# fit_fun = energy.point2point(agg='mean', l=2)
# # regul_fun = energy.grad_disp(l=2)
# regul_fun = energy.alap(transfo='rigid', l=2)
# # reg_defo = register.reg_deformable(niter=50, fit_fun=fit_fun, regul_fun=regul_fun,
# #                                    lr=1e-2, wreg=5e-1, sigma=1e-1, int_steps=16)
# reg_defo = register.reg_deformable(niter=50, fit_fun=fit_fun, regul_fun=regul_fun,
#                                    lr=5e-3, wreg=1e-2, sigma=1e-2, int_steps=16)
# _, mesh_micro_defo, loss = reg_defo.compute(mesh_mri_norm, mesh_micro_aff_norm)
# plt.plot(loss)
# plt.show()
# pts_micro_defo_norm = mesh_micro_defo[0]
# pts_micro_defo = utils.denormalise_pts([pts_micro_defo_norm], mu, amp)[0]
# print(utils.chamfer(pts_micro_defo, pts_mri))
# fig = utils.plot_obj(pts_micro_defo, simps_micro)
# # fig = utils.plot_obj(pts_mri+np.array([100,0,0]), simps_mri, pts_col=(1,0,0), face_col=(1, 0.5,0.5), fig=fig)
# fig = utils.plot_obj(pts_mri, simps_mri, pts_col=(1,0,0), face_col=(1, 0.5,0.5), fig=fig)
# fig.show()


#%%


# slice mri

reg2_rig = register.reg_linear(niter=50, transfo='rigid', init='centroid')
reg2_aff = register.reg_linear(niter=50, transfo='affine')

mesh_micro_moved = utils.polylines_2d_3d(polylines_defo, 2, slice_pos)
mesh_mri_moved = pts_mri_aff, simps_mri, None, None

for _ in range(10):

    polylines_micro_moved = utils.polylines_3d_2d(mesh_micro_moved[:2], dim=2)
    poly_mri_moved = utils.vtkpoly(mesh_mri_moved[0], mesh_mri_moved[1])
    polylines_micro_aff = []
    for i in range(len(slice_pos)):

        pos = slice_pos[i]
        poly_sli = utils.slice_vtkpoly(poly_mri_moved, pos, axis=2)
        if poly_sli.GetPoints() is None:
           polyline_micro_aff = polylines_micro_moved[i][0], polylines_micro_moved[i][1], None, None
        else:
            pts = np.array(poly_sli.GetPoints().GetData())
            pts = np.delete(pts, 2, axis=1)
            nsimps = poly_sli.GetNumberOfLines()
            simps = np.array(poly_sli.GetLines().GetData()).reshape((nsimps,-1))
            simps = np.array(simps, dtype=np.int32).reshape((nsimps,-1))[:,1:]

            polyline_mri = pts, simps, None, None
            polyline_micro = polylines_micro_moved[i][0], polylines_micro_moved[i][1], None, None

            _, polyline_micro_rig = reg2_rig.compute(polyline_mri, polyline_micro)
            _, polyline_micro_aff = reg2_aff.compute(polyline_mri, polyline_micro_rig)

        polylines_micro_aff.append(polyline_micro_aff)

    pts_micro_moved, simps_micro_moved = utils.polylines_2d_3d(polylines_micro_aff, 2 ,slice_pos)
    mesh_micro_moved = pts_micro_moved, simps_micro, None, None

    _, mesh_mri_moved = reg_aff.compute(mesh_micro_moved, mesh_mri_moved)

    print(utils.chamfer(mesh_mri_moved[0], pts_micro_moved))
    fig = utils.plot_obj(mesh_mri_moved[0], simps_mri)
    fig = utils.plot_obj(pts_micro_moved+np.array([0,0,0]), simps_micro_moved, pts_col=(1,0,0), face_col=(1, 0.5,0.5), fig=fig)
    fig.show()

