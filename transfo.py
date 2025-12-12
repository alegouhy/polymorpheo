import jax.numpy as jnp
import numpy as np
from itertools import product
from scipy.linalg import logm
from scipy.linalg import expm
from scipy.stats.qmc import Sobol

import utils


class kernel_disp():
    """
    should investigate: https://github.com/dodgebc/jaxkd
    """
    def __init__(self, sigma='silverman', int_steps=64, int_step_max=None, eps=1e-14):

        self.sigma = sigma
        self.int_steps = int_steps
        self.int_step_max = int_steps if int_step_max is None else int_step_max
        self.eps = eps
        
    def compute(self, pts, cpts, theta_lin=None, theta_trans=None):
        # pts:   (npts, ndims),              points where we want to evaluate the disp.
        # cpts:  (ncpts, ndims),             points where theta is known.              
        # theta_trans: (ncpts, ndims),       transfo params (translation part).
        # theta_lin (ncpts, ndims, ndims),   transfo params (linear part).
        
        if self.sigma == 'silverman':
            self.sigma = self.sigma_silverman(cpts)        
        
        disp = self.interp(pts, cpts, theta_lin, theta_trans)

        if self.int_steps > 0:
            disp = self.lie_exp(disp, pts, cpts, theta_lin, theta_trans)

        return disp

    
    def lie_exp(self, disp, pts, cpts, theta_lin, theta_trans):
        
        disp = disp / self.int_steps
        for i in range(self.int_step_max):
            disp += self.interp(pts + disp, cpts, theta_lin, theta_trans) / self.int_steps

        return disp

    
    def interp(self, pts, cpts, theta_lin, theta_trans):
            
        sqdist = utils.pts_dist(pts, cpts)                                     # (npts, ncpts)
        weight = jnp.exp(-sqdist / (2*self.sigma**2))                            
        weight = weight / (jnp.sum(weight, axis=1)[...,None] + self.eps)

        if theta_trans is not None and theta_lin is None:
            disp =  weight @ theta_trans

        else:
            disp = jnp.zeros_like(pts)  
            if theta_lin is not None:
                disp += jnp.einsum('ij,jkl,il->ik', weight, theta_lin, pts)   
            if theta_trans is not None:
                disp += weight @ theta_trans
        
        return disp
    
        
    def sigma_silverman(self, pts):
        
        iqr = jnp.mean(jnp.quantile(pts, (3/4), axis=0) - jnp.quantile(pts, (1/4),axis=0))
        std = jnp.mean(jnp.std(pts, axis=0))
        sigma = 0.9 * jnp.min(jnp.stack((std, iqr / 1.349))) * pts.shape[0] ** (-1/5)
        
        return sigma
    


class init_transfo():
    
    def __init__(self, init='identity'):
        """
        init: init: 'identity', 'centroids', 'similarity', 'ellipsoid'
        """
        
        self.init = init
    
    def compute(self, ref_pts, mov_pts):
        # init: 'identity', 'centroids', 'similarity', 'ellipsoid'
        
        ndims = ref_pts.shape[1]
    
        if self.init in (None, 'identity'):
            return jnp.eye(ndims+1), mov_pts
        
        else:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
            mov_pts_mu = jnp.mean(mov_pts, axis=0)    
        
            if self.init in ('similarity', 'ellipsoid'):
                s = jnp.std(ref_pts, axis=0) / jnp.std(mov_pts, axis=0)
                if self.init == 'ellipsoid':
                    A = jnp.diag(s)
                elif self.init == 'similarity':
                    A = jnp.mean(s) * jnp.eye(ndims)
            else:
                A = jnp.eye(ndims)
                
            t = ref_pts_mu - (mov_pts_mu @ A.T)
            moved_pts = (mov_pts @ A.T) + t 
            
            T = utils.aff_hmgn(A, t)
                
            return T, moved_pts
    
    
class opti_linear_transfo():
    
    def __init__(self, transfo, se=True): #  gamma=1e-5
        """ 
        transfo: 'rigid' or 'affine'
        """
        
        self.transfo = transfo
        self.se = se
        # self.gamma = gamma
        

    def fit(self, ref_pts, mov_pts,
                  ref_pts_mu=None, mov_pts_mu=None):
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """

        ndims = ref_pts.shape[1]
        
        if ref_pts_mu is None:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.mean(mov_pts, axis=0)
        self.ref_pts_mu = ref_pts_mu
        self.mov_pts_mu = mov_pts_mu
        ref_pts_bar = ref_pts - ref_pts_mu
        mov_pts_bar = mov_pts - mov_pts_mu
            
        # nu = jnp.unique(ref_pts_bar, axis=0).shape[0]
        # if nu <= ndims + 2:   # Tikhonov regularization
        #     mov_pts_bar = jnp.vstack([mov_pts_bar, self.gamma * jnp.eye(ndims)])
        #     ref_pts_bar = jnp.vstack([ref_pts_bar, jnp.zeros((ndims, ndims))])
        
        if self.transfo in ('rigid', 'similarity'):   
            cov = ref_pts_bar.T @ mov_pts_bar
            U, _, Vt = jnp.linalg.svd(cov, full_matrices=False)
            S = jnp.eye(ndims)
            
        if self.transfo == 'similarity':
            mov_norm_sq = jnp.sum(mov_pts_bar ** 2)
            s = jnp.trace(cov @ Vt.T @ U.T) / mov_norm_sq
            S = s * S
 
        elif self.transfo == 'affine':
            A, _, _, _ = jnp.linalg.lstsq(mov_pts_bar, ref_pts_bar, rcond=None)
            U, S, Vt = jnp.linalg.svd(A.T, full_matrices=False)                      
            S = jnp.diag(S)

        if self.se:
            det_sign = jnp.sign(jnp.linalg.det(U @ Vt))
            S = S.at[-1,-1].set(S[-1,-1]*det_sign)

        A = (U @ S) @ Vt
         
        t = ref_pts_mu - (mov_pts_mu @ A.T)

        return A, t


    def transform(self, A, t, mov_pts):
        
        return (mov_pts @ A.T) + t
    
    

class opti_polynom_transfo():
    
    def __init__(self, degree, se=True):

        self.degree = degree
    

    def fit(self, ref_pts, mov_pts,
                  ref_pts_mu=None, mov_pts_mu=None):
        """
        Assumes that ref_pts and mov_pts are paired sets of points.
        """
        
        if ref_pts_mu is None:
            ref_pts_mu = jnp.mean(ref_pts, axis=0)
        if mov_pts_mu is None:
            mov_pts_mu = jnp.mean(mov_pts, axis=0)
        self.ref_pts_mu = ref_pts_mu
        self.mov_pts_mu = mov_pts_mu
        ref_pts_bar = ref_pts - ref_pts_mu
        mov_pts_bar = mov_pts - mov_pts_mu         
        
        X, _ = self.design_mat(mov_pts_bar)
        
        coeffs, _, _, _ = jnp.linalg.lstsq(X, ref_pts_bar, rcond=None)

        return coeffs
    
    
    def design_mat(self, x):
        
        npts, ndims = x.shape
        exponents = []
        for total_deg in range(self.degree + 1):
            for exponent in product(range(total_deg + 1), repeat=ndims):
                if sum(exponent) == total_deg:
                    exponents.append(exponent)
                    
        X = jnp.ones((npts, len(exponents)))
        for j, e in enumerate(exponents):
            X = X.at[:, j].set(jnp.prod(x ** jnp.array(e), axis=1))
            
        return X, exponents
    
    
    def transform(self, coeffs, mov_pts, X=None):
        
        mov_pts_bar = mov_pts - self.mov_pts_mu
        
        if X is None:
            X, _ = self.design_mat(mov_pts_bar)
                    
        return X @ coeffs + self.ref_pts_mu
        
    
    
def random_locAff(ndims, ncpts=300, seed=None, liealg=True,
                  so_corner=[0], ne_corner=[1], bound_scal=1.1,
                  trans_bounds=0.01, rot_bounds=np.pi/2, scalDir_bounds=np.pi/4, scal_bounds=3):
              
    if len(so_corner) == 1: so_corner *= ndims 
    if len(ne_corner) == 1: ne_corner *= ndims 
    so_corner = np.array(so_corner)
    ne_corner = np.array(ne_corner)
    
    center = (ne_corner + so_corner) / 2 
    ne_corner = np.expand_dims(bound_scal * (ne_corner - center) + center, axis=-1)
    so_corner = np.expand_dims(bound_scal * (so_corner - center) + center, axis=-1)
    
    r = Sobol(ndims, seed=seed).random(ncpts)
    cpts = np.expand_dims(r*ne_corner.T + (1-r)*so_corner.T, axis=-1)
    
    if ndims == 2:
        rot_angles = 2*rot_bounds*(np.random.rand(ncpts)-0.5)
        rot = np.stack([np.stack([ np.zeros(ncpts), rot_angles ], axis=1),
                        np.stack([-rot_angles , np.zeros(ncpts)], axis=1)], axis=2)   
    
        scalDir_angles = 2*scalDir_bounds*(np.random.rand(ncpts)-0.5)
        scalDir = np.stack([np.stack([ np.zeros(ncpts)   , scalDir_angles], axis=1),
                            np.stack([-scalDir_angles, np.zeros(ncpts)   ], axis=1)], axis=2)   
        
    elif ndims == 3:
        rot_angles = 2*rot_bounds*(np.random.rand(ncpts,3)-0.5) 
        rot = np.stack([np.stack([ np.zeros(ncpts),  rot_angles[...,2], -rot_angles[...,1]], axis=1),
                        np.stack([-rot_angles[...,2],  np.zeros(ncpts),  rot_angles[...,0]], axis=1),
                        np.stack([ rot_angles[...,1], -rot_angles[...,0],  np.zeros(ncpts)], axis=1)], axis=2)
    
        scalDir_angles = 2*scalDir_bounds*(np.random.rand(ncpts,3)-0.5) 
        scalDir = np.stack([np.stack([ np.zeros(ncpts)    ,  scalDir_angles[...,2], -scalDir_angles[...,1]], axis=1),
                            np.stack([-scalDir_angles[...,2],  np.zeros(ncpts)    ,  scalDir_angles[...,0]], axis=1),
                            np.stack([ scalDir_angles[...,1], -scalDir_angles[...,0],  np.zeros(ncpts)    ], axis=1)], axis=2)
    
    rot = expm(rot)
    scalDir = expm(scalDir)
    
    scal_factors = np.exp(2*np.log(scal_bounds)*(np.random.rand(ncpts, ndims, 1)-0.5))
    scal = np.eye(ndims) * scal_factors
    
    mats = np.matmul(rot,np.matmul(scalDir,np.matmul(scal,np.transpose(scalDir, [0,2,1]))))
    trans = 2*trans_bounds*(np.random.rand(ncpts,ndims,1)-0.5)
    trans = np.matmul(mats, -cpts) + trans + cpts
    
    if not liealg:
        return cpts[..., 0], mats, trans[...,0]
    
    affs = np.concatenate((mats, trans), axis=2)
    affs = np.concatenate((affs, np.concatenate((np.zeros((ncpts,1,ndims)), np.ones((ncpts,1,1))), axis=2)), axis=1)
    
    log_affs = np.stack([logm(affs[i,...]) for i in range(ncpts)], axis=0)
    log_mats = log_affs[:, :ndims, :ndims]
    log_trans = log_affs[:, :ndims, ndims]
    
    return cpts[..., 0], log_mats.real, log_trans.real