import os
os.chdir('/Users/alegouhy/tests/contours2mesh')
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import matplotlib.pyplot as plt
import time

import utils
import energy
import register

#%%

mov_file = 'imgs/mov.png'
ref_file = 'imgs/ref.png'

mov_img = utils.load_img(mov_file)
ref_img = utils.load_img(ref_file)

npts = 100
get_normals = True
mov_contour = utils.seg_to_contour(mov_img, npts=npts, get_normals=get_normals)
ref_contour = utils.seg_to_contour(ref_img, npts=npts, get_normals=get_normals)
mov_pts, mov_simps, mov_normals, mov_labs = mov_contour
ref_pts, ref_simps, ref_normals, ref_labs = ref_contour
[mov_pts, ref_pts], mean, std = utils.normalise_pts([mov_pts, ref_pts])
mov_contour = mov_pts, mov_simps, mov_normals, mov_labs
ref_contour = ref_pts, ref_simps, ref_normals, ref_labs

plt.subplot(1,2,1)
utils.plot_img(mov_img)
plt.subplot(1,2,2)
utils.plot_img(ref_img)
plt.show()

plt.subplot(1,2,1)
utils.plot_contour(mov_contour, col=[1,0,0])
plt.subplot(1,2,2)
utils.plot_contour(ref_contour, col=[0,0,1])
plt.show()

utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(mov_contour, col=[0,0,1])
plt.show()


#%% rigid ICP

niter = 20
transfo = 'rigid'
init = 'centroid'

reg_rig = register.reg_linear(niter=niter, transfo=transfo, init=init, se=True, plot=False)

t = time.time()
rig, moved_contour = reg_rig.compute(ref_contour, mov_contour)

print(f"rigid done in: {time.time()-t:.2f} s")

utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour, col=[0,0,1])
plt.title('rigid')
plt.show()

#%% affine ICP

niter = 20
transfo = 'similarity' # 'affine'

reg_aff = register.reg_linear(niter=niter, transfo=transfo, se=True, plot=False)

t = time.time()
aff, moved_contour = reg_aff.compute(ref_contour, mov_contour, T0=rig)
print(f"affine done in: {time.time()-t:.2f} s")

utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour, col=[0,0,1])
plt.title('affine')
plt.show()


#%% quadratic ICP

niter = 20
degree = 2

reg_poly = register.reg_polynom(niter=niter, degree=degree, init='identity', se=True, plot=False)

t = time.time()
moved_contour = reg_poly.compute(ref_contour, mov_contour)
print(f"poly done in: {time.time()-t:.2f} s")

utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour, col=[0,0,1])
plt.title('quadratic')
plt.show()


#%% quintic ICP

niter = 20
degree = 5

reg_poly = register.reg_polynom(niter=niter, degree=degree, init='identity', se=True, plot=False)

t = time.time()
moved_contour = reg_poly.compute(ref_contour, mov_contour)
print(f"poly done in: {time.time()-t:.2f} s")

utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour, col=[0,0,1])
plt.title('quintic')
plt.show()


#%% non-rigid ICP

niter = 50
# lr = [1e-2, 1e-3, 1e-4]
# wreg = [1e-1, 1e-2, 1e-4]
lr = [1e-2, 1e-2]
wreg = [1e-1, 1e-2] # [1e-1]
int_steps = 16
sigma = None # 1e-3 # 'silverman'

fit_fun = energy.pointdist(dist='chamfer', normtype='l1')

# regul_fun =  energy.grad_disp(normtype='l2')
regul_fun =  energy.alap(transfo='similarity', normtype='l2')
neighs = utils.neighs_contour(mov_simps, mov_pts.shape[0])
regul_fun.set_neighs(neighs)

disp = utils.aff_mat2disp(aff, mov_pts)

moved_contour = mov_pts + disp, mov_simps, mov_normals
utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour, col=[0,0,1])
plt.show()

losses = []
t = time.time()
for i in range(len(lr)):
    
    reg_defo = register.reg_deformable(niter=niter, fit_fun=fit_fun, regul_fun=regul_fun,
                                       lr=lr[i], wreg=wreg[i], sigma=sigma, int_steps=int_steps, plot=1)
    
    disp, moved_contour, loss = reg_defo.compute(ref_contour, mov_contour, disp0=disp)
    
    losses.append(loss)

print(f"deformable done in: {time.time()-t:.2f} s")

utils.plot_contour(ref_contour, col=[1,0,0])
utils.plot_contour(moved_contour, col=[0,0,1])
plt.show()

for i in range(len(lr)):
    plt.subplot(1,len(lr),i+1)
    plt.plot(losses[i])
    plt.suptitle('energy')
plt.show()
