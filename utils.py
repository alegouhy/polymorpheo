import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
import jax.numpy as jnp
import jax
from scipy.linalg import logm    # not in jax.scipy yet...
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy, numpy_to_vtkIdTypeArray
from shapely.geometry import LinearRing


def develop(x, dec='|_'):
    
    if isinstance(x, (list, tuple)):
        print(dec, type(x), len(x))
        dec = '|    ' + dec
        for i in range(len(x)):
            develop(x[i], dec)
    elif isinstance(x, np.ndarray):
        print(dec, type(x), x.dtype, x.shape)   
    else: 
        print(dec, type(x))   
        
        
def load_img(img_file):
    
    img = skimage.io.imread(img_file)
    
    return img[..., 0]


def opts_to_contour(opts_list, npts=None, get_simps=True, get_normals=False):
    """
    list of ordered points to contour
    """
    
    normals = None
    pts = []
    simps = [] if get_simps else None
    ipt = 0
    
    if npts is not None:
        lengths = []
        for i, opts in enumerate(opts_list):
            if opts.shape[0] < 2: length = lengths.append(0.0)
            else: length = np.sum(np.linalg.norm(np.diff(opts, axis=0), axis=1))
            lengths.append(length)
        length_tot = np.sum(lengths)
        npts_contour = [int(npts * length / length_tot) for length in lengths[:-1]]
        npts_contour += [int(npts - np.sum(npts_contour))]
        
    for i, opts in enumerate(opts_list):
        
        if npts is not None:
            opts = resample_contour(opts, npts_contour[i])
        # if decim != 1:
        #     opts = opts[::decim]
        n_pts = opts.shape[0]
        pts.append(opts)
        
        if get_simps:
            indices = np.arange(ipt, ipt + n_pts)
            edges = np.stack([indices[:-1], indices[1:]], axis=1)
            edges = np.concatenate((edges, [[ipt + n_pts - 1, ipt]]), axis=0)
            simps.append(edges)
        
        ipt += n_pts
    
    pts = np.array(np.concatenate(pts, axis=0))
    if get_simps: 
        simps = np.array(np.concatenate(simps, axis=0) )
    if get_normals:
        normals = normals_contour(pts, simps)

    return pts, simps, normals
   

def close_contours(opts):
    
    if np.all(opts[0,:] != opts[-1,:]):
        np.vstack((opts, opts[0,:]))
        
    return opts


def split_opts(opts, i, j):
    # with i < j
    
    opts1 = np.vstack((opts[:i+1, :], opts[j:-1, :]))
    opts2 = np.vstack((opts[i:j+1, :]))
    
    return [opts1, opts2]


def splitfit_opts(opts_1, opts_2):

    i_hat = 0
    j_hat = 0
    dist_hat = np.inf
    
    n = opts_1[0].shape[0]
    
    for i in range(n):
        for j in range(n):
            if i > j: continue
            
            opts_1_split = split_opts(opts_1[0], i, j)
            
            dist1 = chamfer(opts_1_split[0], opts_2[0]) + \
                    chamfer(opts_1_split[1], opts_2[1])
            
            dist2 = chamfer(opts_1_split[0], opts_2[1]) + \
                    chamfer(opts_1_split[1], opts_2[0])
            
            dist = min(dist1, dist2)
            
            if dist < dist_hat:
                i_hat = i
                j_hat = j
                dist_hat = dist
                swap = True if dist2 < dist1 else False
    
    opts_1_split = split_opts(opts_1[0], i_hat, j_hat)
    opts_1_split = cw(opts_1_split)
    if swap:
        opts_1_split = [opts_1_split[1], opts_1_split[0]]
    
    return opts_1_split


def seg_to_contour(seg, npts=None, get_simps=True, get_normals=False):

    opts_list = skimage.measure.find_contours(seg > 0)
    
    opts_list = [opts[:,[1,0]] for opts in opts_list]
    
    pts, simps, normals = opts_to_contour(opts_list, npts=npts, get_simps=get_simps, get_normals=get_normals)

    return pts, simps, normals


@jax.jit
def pts_sqdist(pts1, pts2):
    
    diff = pts1[:,None,:] - pts2[None,:,:]
    
    return jnp.sum(diff ** 2, axis=-1)


def chamfer(pts1, pts2):
    
    dist = pts_sqdist(pts1, pts2)
    
    dist = jnp.mean(jnp.min(dist, axis=0))\
          + jnp.mean(jnp.min(dist, axis=1))
    
    return dist
    

def cw(opts):
    
    opts_cw = []
    for opt in opts:
        opt_cw = opt
        if opt.shape[0] > 3:
            if LinearRing(opt).is_ccw == True:
                opt_cw = np.flipud(opt)
        opts_cw.append(opt_cw)
      
    return opts_cw
    

def aff_hmgn(lin, trans):
    
    ndims = lin.shape[0]
    
    M = jnp.eye(ndims+1)
    M = M.at[:ndims, :ndims].set(lin)
    M = M.at[:ndims, ndims].set(trans)
    
    return M

def aff_dehmgn(M):
    
    ndims = M.shape[1] - 1
    
    lin = M[:ndims, :ndims]
    trans = M[:ndims, ndims]
    
    return lin, trans


def aff_mat2disp(M, mov_pts, do_log=False):
    
    if do_log:
        M = logm(M)
        
    lin, trans = aff_dehmgn(M)
    moved_pts =  (mov_pts @ lin.T) + trans
    
    return moved_pts - mov_pts


def simps_to_adj(simps, npts=None):
    
    ndims = simps.shape[1]
    if npts is None:
        npts = np.max(simps) + 1
        
    adj = np.zeros((npts, npts), dtype=int)
    adj[simps[:,0], simps[:,1]] = 1
    if ndims == 3:
        adj[simps[:,1], simps[:,2]] = 1
        adj[simps[:,2], simps[:,0]] = 1
    adj = adj + adj.T

    return adj


def normals_contour(pts, simps, eps=1e-9):

    edges =  pts[simps[:, 1]] -  pts[simps[:, 0]]
                                     
    edge_normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1)
    # edge_normals = edge_normals / (jnp.linalg.norm(edge_normals, axis=1, keepdims=True) + eps)
    
    pts_normals = np.zeros(pts.shape)
    # pts_normals = pts_normals.at[simps[:, 0]].add(edge_normals)
    # pts_normals = pts_normals.at[simps[:, 1]].add(edge_normals)
    pts_normals[simps[:, 0]] += edge_normals
    pts_normals[simps[:, 1]] += edge_normals
    
    pts_normals = pts_normals / (np.linalg.norm(pts_normals, axis=1, keepdims=True) + eps)
    
    return pts_normals


def concat_contours(contour_list):
    
    is_simps = contour_list[0][1] is not None
    is_normals = contour_list[0][2] is not None
    
    pts = []
    simps = [] if is_simps else None
    normals = [] if is_normals else None
    offset = 0
    for contour in contour_list:
        
        pts.append(contour[0])
        if is_simps:
            simps.append(contour[1] + offset)
        if is_normals:
            normals.append(contour[1])
            
        offset += contour[0].shape[0]
        
    pts = np.concatenate(pts, axis=0)
    if is_simps:
        simps = np.concatenate(simps, axis=0)
    if is_normals:
        normals = np.concatenate(normals, axis=0)
        
    return pts, simps, normals


def neighs_contour(simps, npts=None):
    
    if npts is None:
        npts = jnp.max(simps) + 1
        
    neighs = jnp.zeros((npts, 3), simps.dtype)
    for i in range(npts):
        neigh = simps[jnp.where(simps == i)[0], :].ravel()
        neigh = jnp.unique(neigh)
        neighs = neighs.at[i,:].set(neigh)
        
    return neighs

            
def normalise_pts(pts_list, mean=None, std=None):
    
    pts_all = np.concatenate(pts_list, axis=0)
    
    if mean is None: mean = np.mean(pts_all, axis=0)
    if std is None: std = np.std(pts_all, axis=0)
    
    pts_norm = [(pts - mean) / std for pts in pts_list]
    
    return pts_norm, mean, std


def plot_img(img):
    
    plt.imshow(img, origin='lower')    
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    
    
def plot_contour(contour,
                 col=[1,0,0], linewidth=1, markersize=6,
                 xlim=[-2,2], ylim=[-2,2], scal_normals=0.1):
    
    pts, simps, normals = contour
    npts, ndims = pts.shape
    col_pts = np.array(col)
    col_simps = (1 + col_pts) / 2

    if simps is not None: 
        for edge in simps:
            plt.plot(pts[edge, 0], pts[edge, 1], '-', color=col_simps, linewidth=linewidth, zorder=-2)
        
    if normals is not None:
        normals = scal_normals * normals
        for i in range(npts):
            plt.plot([pts[i,0], pts[i,0]+normals[i,0]], [pts[i,1], pts[i,1]+normals[i,1]], color=col_simps)
            
    plt.scatter(pts[1:-1,0], pts[1:-1,1], markersize, color=col_pts) 
    plt.scatter(pts[0,0], pts[0,1], 4*markersize, color=col_pts, marker='^')
    plt.scatter(pts[-1,0], pts[-1,1], 3*markersize, color=col_pts, marker='s')
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    


def plot_opts(opts,
              col=[1,0,0], linewidth=1, markersize=6,
              xlim=[-2,2], ylim=[-2,2], scal_normals=0.1):
    
    col_pts = np.array(col)
    
    plt.plot(opts[:,0], opts[:,1], color=col_pts) 
    plt.scatter(opts[1:-1,0], opts[1:-1,1], markersize, color=col_pts) 
    plt.scatter(opts[0,0], opts[0,1], 4*markersize, color=col_pts, marker='^')
    plt.scatter(opts[-1,0], opts[-1,1], 3*markersize, color=col_pts, marker='s')
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)
    
    
    
def plot_disp(disp, pts,
              col=[1,0,0], linewidth=1, markersize=3,
              xlim=[-2,2], ylim=[-2,2], scal_normals=0.1):
    
    col_pts = np.array(col)
    col_vecs = (1 + col_pts) / 2

    plt.quiver(pts[:,0], pts[:,1], disp[:,0], disp[:,1], color=col_vecs)
    
    plt.axis('off')
    plt.gca().set_aspect('equal')
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)


def resample_contour(pts, n):
    """
    assumes 2D pts, ordered
    """

    ndims = pts.shape[1]
    
    seq = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    seq /= seq[-1]
    
    seq_res = np.linspace(0, 1, n, endpoint=False)
    
    pts_res = [np.interp(seq_res, seq, pts[:, d]) for d in range(ndims)]

    return np.stack(pts_res, axis=-1)


    
# def phase_align_contours(opts1, opts2, simps2=None):
#     """
#     assumes 2D pts, ordered and npts1 = npts2
#     """

#     npts = opts1.shape[0]
#     best_k = 0
#     best_dist = np.inf
    
#     for k in range(npts):
#         opts2_k = np.roll(opts2, shift=k, axis=0)
#         dist = np.sum(np.linalg.norm(opts1 - opts2_k, axis=1))
#         if dist < best_dist:
#             best_dist = dist
#             best_k = k
    
#     opts2 = np.roll(opts2, shift=best_k, axis=0)
    
#     if simps2 is None:
#         return opts2
#     else: 
#         simps2 = (simps2 + best_k) % npts
#         return opts2, simps2
    

def bridge_contours(pts_list, z_coords, npts=None):
    """
    assumes 3D pts are ordered
    
    """
    
    simps = []
    pts = []
    if npts is None:
        npts = pts_list[0].shape[0]
        do_res = False
    else: 
        do_res= True
     
    for k in range(len(pts_list)-1):
        offset = 2 * npts * k
        
        pts1 = pts_list[k]
        pts2 = pts_list[k+1]
        z1 = z_coords[k]
        z2 = z_coords[k+1]
        
        if do_res:
            pts1 = resample_contour(pts1, npts)
            pts2 = resample_contour(pts2, npts)
        pts2 = phase_align_contours(pts1, pts2)
        
        pts1 = np.c_[pts1, np.full(npts, z1)]
        pts2 = np.c_[pts2, np.full(npts, z2)]
        
        pts.append(np.vstack([pts1, pts2]))

        for i in range(npts):
            i1, i2 = i, (i + 1) % npts
            j1, j2 = npts + i, npts + ((i + 1) % npts)
            simps.append(np.array([i1, j1, j2], dtype=int) + offset)
            simps.append(np.array([i1, j2, i2], dtype=int) + offset)

    pts = np.concatenate(pts, axis=0)
    simps = np.vstack(simps).astype(int)
    
    return pts, simps


def bridge_contours_2(opts_list, z_coords, greedy=False, sealed=True):
    
    pts = []
    simps = []
    offset = 0
    if greedy: 
        path_fun = triangle_path_greedy
    else:
        path_fun = triangle_path_dp
    
    if sealed:
        lid_start = [np.mean(opts, axis=0)[None,...] for opts in opts_list[0]]
        lid_end = [np.mean(opts, axis=0)[None,...] for opts in opts_list[-1]]
        opts_list = [lid_start] + opts_list + [lid_end]
        z_coords = np.hstack((z_coords[0], z_coords, z_coords[-1]))

    for i in range(len(opts_list)-1):
        print(i)

        opts_1 = cw(opts_list[i])
        opts_2 = cw(opts_list[i+1])
        
        n1 = len(opts_1)
        n2 = len(opts_2)
        if n1 == 1 and n2 == 2:
            opts_1 = splitfit_opts(opts_1, opts_2)
            n1 += 1
        elif n1 == 2 and n2 == 1:
            opts_2 = splitfit_opts(opts_2, opts_1)
            n2 += 1
        elif n1 != n2:
            raise ValueError(f"Can't handle {n1} to {n2} branching yet")
        
        for k in range(n1):
            
            opts_1k = close_contours(opts_1[k])
            opts_2k = close_contours(opts_2[k])
            nn1 = opts_1k.shape[0]
            nn2 = opts_2k.shape[0]
            
            opts_2k = phase_align_contours(opts_1k, opts_2k)
            
            plt.subplot(2,n1,k+1)
            plot_opts(opts_1k, col=[1,0,0])
            plot_opts(opts_2k, col=[0,0,1])
            plt.title(str(k))
            
            opts_1k = np.hstack([opts_1k, np.full((opts_1k.shape[0],1), z_coords[i])])
            opts_2k = np.hstack([opts_2k, np.full((opts_2k.shape[0],1), z_coords[i+1])])
            
            pts_k = np.vstack([opts_1k, opts_2k])
             
            path, path_len = path_fun(opts_1k, opts_2k)
            simps_k = triangulate_path(path, path_len, nn1, nn2)
            
            plt.subplot(2,n1,n1+k+1)
            plot_path(path, nn1, nn2)
            
            simps_k = simps_k + offset
            
            pts.append(pts_k)
            simps.append(simps_k)
            
            offset += pts_k.shape[0]
        
        plt.suptitle(str(i))
        plt.show()
                 
    pts = np.vstack(pts)
    simps = np.vstack(simps)
    
    return pts, simps
    

def phase_align_contours(opts1, opts2, simps2=None, start=True, end=True):

    npts = opts1.shape[0]
    
    opts1_end = []
    if start: opts1_end.append(opts1[0,:])
    if end: opts1_end.append(opts1[-1,:])
    opts1_end = np.vstack(opts1_end)
    
    dist = pts_sqdist(opts1_end, opts2)
    dist = np.sum(dist, axis=0)
    
    k = np.argmin(dist)
    opts2 = np.roll(opts2, shift=-k, axis=0)
    
    if simps2 is None:
        return opts2
    else: 
        simps2 = (simps2 + k) % npts
        return opts2, simps2


# def triangulate_path(path, n1, n2):

#     simps = []
    
#     for k in range(len(path) - 1):
#         i_curr, j_curr = path[k]
#         i_next, j_next = path[k + 1]
        
#         di = i_next - i_curr
#         dj = j_next - j_curr
        
#         if di == 1 and dj == 0:
#             simps.append([i_curr, i_next, n1 + j_curr])
            
#         elif di == 0 and dj == 1:
#             simps.append([i_curr, n1 + j_curr, n1 + j_next])
            
#         elif di == 1 and dj == 1:
#             simps.append([i_curr, i_next, n1 + j_next])
#             simps.append([i_curr, n1 + j_curr, n1 + j_next])
            
#     simps.append([n1 - 1, 0, n1 + n2 - 1])
#     simps.append([0, n1, n1 + n2 - 1])

#     return jnp.array(simps)




def triangulate_path(path, path_length, n1, n2):
    simps = []
    
    # Only iterate over valid path entries
    for k in range(path_length - 1):  # path_length is now a Python int after JIT
        i_curr, j_curr = int(path[k, 0]), int(path[k, 1])
        i_next, j_next = int(path[k + 1, 0]), int(path[k + 1, 1])
        
        di = i_next - i_curr
        dj = j_next - j_curr
        
        if di == 1 and dj == 0:
            simps.append([i_curr, i_next, n1 + j_curr])
            
        elif di == 0 and dj == 1:
            simps.append([i_curr, n1 + j_curr, n1 + j_next])
            
        elif di == 1 and dj == 1:
            simps.append([i_curr, i_next, n1 + j_next])
            simps.append([i_curr, n1 + j_curr, n1 + j_next])
            
    simps.append([n1 - 1, 0, n1 + n2 - 1])
    simps.append([0, n1, n1 + n2 - 1])

    return jnp.array(simps)



def triangle_path_greedy(opts1, opts2):
    
    n1, n2 = opts1.shape[0], opts2.shape[0]
    
    path = [(0, 0)]
    i, j = 0, 0
    
    while i < n1 - 1 or j < n2 - 1:
        
        if i >= n1 - 1:
            j += 1
            
        elif j >= n2 - 1:
            i += 1
            
        else:
            dist1 = jnp.sum((opts1[i+1] - opts2[j]) ** 2)
            dist2 = jnp.sum((opts1[i] - opts2[j+1]) ** 2)
            
            if dist1 < dist2:
                i += 1
            else:
                j += 1
        
        path.append((i, j))
    
    return path


def plot_path(path, n1, n2):
    
    mat = np.zeros((n1, n2))
    
    for p in path:
        mat[p[0], p[1]] = 1
    
    plt.imshow(mat)
    plt.axis('off')
  
    
@jax.jit
def triangle_path_dp(opts1, opts2):
    
    opts1 = jnp.array(opts1)
    opts2 = jnp.array(opts2)
    n1, n2 = opts1.shape[0], opts2.shape[0]
    
    dists = jnp.linalg.norm(opts1[:, None, :] - opts2[None, :, :], axis=2)
    
    dp = jnp.full((n1, n2), jnp.inf)
    dp = dp.at[0, 0].set(0)
    parent = jnp.zeros((n1, n2), dtype=jnp.int32)
    
    def body_fn(ij, carry):
        dp, parent = carry
        i = ij // n2
        j = ij % n2
        
        skip = (i == 0) & (j == 0)
        
        costs = jnp.array([jnp.where(j > 0, dp[i, j-1] + dists[i, j], jnp.inf),
                           jnp.where(i > 0, dp[i-1, j] + dists[i, j], jnp.inf),
                           jnp.where((i > 0) & (j > 0), dp[i-1, j-1] + dists[i, j] + dists[i-1, j-1], jnp.inf)])
        
        ind_best = jnp.argmin(costs)
        cost_best = costs[ind_best]
        
        dp = jnp.where(skip, dp, dp.at[i, j].set(cost_best))
        parent = jnp.where(skip, parent, parent.at[i, j].set(ind_best))
        
        return dp, parent
    
    dp, parent = jax.lax.fori_loop(0, n1 * n2, body_fn, (dp, parent))
    
    # Backtrack
    max_path_len = n1 + n2
    
    def backtrack_cond(state):
        i, j, step, path = state
        at_start = (i == 0) & (j == 0)
        return ~at_start & (step < max_path_len - 1)
    
    def backtrack_body(state):
        i, j, step, path = state
        path = path.at[step].set(jnp.array([i, j]))
        
        direction = parent[i, j]
        i_new = jnp.where(direction == 0, i, i - 1)
        j_new = jnp.where(direction == 1, j, j - 1)
        
        return i_new, j_new, step + 1, path
    
    path = jnp.zeros((max_path_len, 2), dtype=jnp.int32)
    init_state = (n1 - 1, n2 - 1, 0, path)
    i_final, j_final, final_step, path = jax.lax.while_loop(backtrack_cond, backtrack_body, init_state)

    path = path.at[final_step].set(jnp.array([0, 0]))
    path_length = final_step + 1
    
    path = jnp.flip(path, axis=0)
    
    # Shift so valid data starts at index 0
    shift_amount = max_path_len - path_length
    path = jnp.roll(path, -shift_amount, axis=0)
    
    return path, path_length

# def triangle_path_dp(opts1, opts2):
    
#     opts1 = jnp.array(opts1)
#     opts2 = jnp.array(opts2)
#     n1, n2 = opts1.shape[0], opts2.shape[0]
    
#     dp = jnp.full((n1, n2), np.inf)
#     dp = dp.at[0, 0].set(0)
#     parent = jnp.zeros((n1, n2), dtype=int)
    
#     for i in range(n1):
#         for j in range(n2):
#             if i == 0 and j == 0: continue
            
#             candidates = jnp.full(3, np.inf)
            
#             # Option 0: step along opts2 from (i, j-1)
#             if j > 0:
#                 edge_cost = jnp.linalg.norm(opts1[i] - opts2[j])
#                 candidates = candidates.at[0].set(dp[i, j-1] + edge_cost)
            
#             # Option 1: step along opts1 from (i-1, j)
#             if i > 0:
#                 edge_cost = jnp.linalg.norm(opts1[i] - opts2[j])
#                 candidates = candidates.at[1].set(dp[i-1, j] + edge_cost)
            
#             # Option 2: diagonal from (i-1, j-1)
#             if i > 0 and j > 0:
#                 edge_cost = (jnp.linalg.norm(opts1[i] - opts2[j]) + 
#                              jnp.linalg.norm(opts1[i-1] - opts2[j-1]))
#                 candidates = candidates.at[2].set(dp[i-1, j-1] + edge_cost)
            
#             ind = jnp.argmin(candidates)
#             dp = dp.at[i, j].set(candidates[ind])
#             parent = parent.at[i, j].set(ind)  # Store 0, 1, or 2
    
#     # Backtrack to get path
#     path = []
#     i, j = n1 - 1, n2 - 1
    
#     while i >= 0 and j >= 0:
#         path.append((i, j))
        
#         if i == 0 and j == 0:
#             break
        
#         direction = parent[i, j]
#         if direction == 2:
#             i -= 1
#             j -= 1
#         elif direction == 0:
#             j -= 1
#         elif direction == 1:
#             i -= 1
#         else:
#             break
    
#     path.reverse()
#     return path


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
    
    return pts, simps, normals



def triangulate_contours_greedy(c1, c2):

    n1, n2 = c1.shape[0], c2.shape[0]
    faces = []
    
    i, j = 0, 0
    
    dp = np.zeros((n1, n2), dtype=int)   ######
    dp[0, 0] = 1
    
    while i < n1 - 1 or j < n2 - 1:
        if i >= n1 - 1:
            # Only can advance along c2
            faces.append([i, n1 + j, n1 + j + 1])
            j += 1
        elif j >= n2 - 1:
            # Only can advance along c1
            faces.append([i, i + 1, n1 + j])
            i += 1
        else:
            # Choose direction based on edge lengths
            # Option 1: advance along c1
            dist1 = np.linalg.norm(c1[i+1] - c2[j])
            # Option 2: advance along c2
            dist2 = np.linalg.norm(c1[i] - c2[j+1])
            
            if dist1 < dist2:
                # Advance along c1: triangle [c1[i], c1[i+1], c2[j]]
                faces.append([i, i + 1, n1 + j])
                i += 1
            else:
                # Advance along c2: triangle [c1[i], c2[j], c2[j+1]]
                faces.append([i, n1 + j, n1 + j + 1])
                j += 1
            
        dp[i,j] = 1  ######
    plt.subplot(1,2,1)
    plt.imshow(dp)
            
    return np.array(faces)



from collections import defaultdict
def contours2opts(pts, simps, closed_only=True):

    adj = defaultdict(list)
    for a, b in simps:
        adj[a].append(b)
        adj[b].append(a)

    # track visited edges
    visited = set()
    contours = []

    for start in range(len(pts)):
        # find edges starting at this node
        neighbors = adj[start]
        if not neighbors:
            continue

        for nb in neighbors:
            edge = tuple(sorted((start, nb)))
            if edge in visited:
                continue

            contour_idx = [start]
            prev, cur = start, nb
            visited.add(edge)

            # follow chain
            while True:
                contour_idx.append(cur)
                nbrs = [n for n in adj[cur] if n != prev]
                if not nbrs:
                    break  # open end
                nxt = nbrs[0]
                edge = tuple(sorted((cur, nxt)))
                if edge in visited:
                    # if we reached the start, it's closed
                    if nxt == contour_idx[0]:
                        contour_idx.append(nxt)
                    break
                visited.add(edge)
                prev, cur = cur, nxt

            # only add closed contours unless otherwise requested
            if (not closed_only) or (contour_idx[0] == contour_idx[-1]):
                contours.append(pts[np.array(contour_idx)])

    return contours


def smooth_vtkpoly(poly, niter=300):
    
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetNumberOfIterations(niter)
    smoother.Update()  
    
    return smoother.GetOutput()


def vtkpoly(pts, simps):
    
    ndims = pts.shape[1]
    
    pts_vtk = vtk.vtkPoints()
    pts_vtk.SetData(numpy_to_vtk(pts, deep=True))
    
    flat_simps = np.hstack([np.full((simps.shape[0], 1), ndims), simps]).flatten()
    simps_vtk = vtk.vtkCellArray()
    simps_vtk.SetCells(simps.shape[0], numpy_to_vtkIdTypeArray(flat_simps, deep=True))

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts_vtk)
    poly.SetPolys(simps_vtk)

    return poly


def write_vtkpoly(poly, filename):
    
    _, ext = os.path.splitext(filename)
    
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetCompressorTypeToZLib()
    elif ext == '.obj':
        writer = vtk.vtkOBJWriter()
        
    writer.SetInputData(poly)
    writer.SetFileName(filename)
    writer.Write()


def read_vtkpoly(filename):

    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.ply':
        reader = vtk.vtkPLYReader()
    elif ext == '.obj':
        reader = vtk.vtkOBJReader()

    reader.SetFileName(filename)
    reader.Update()
    poly = reader.GetOutput()
    
    return poly


def render_vtkpoly(polydatas, wireframe=False):
    
    n = len(polydatas)
    cols = plt.get_cmap('tab10_r')(np.arange(n))
    
    mappers = [vtk.vtkPolyDataMapper() for i in range(n)]
    actors = [vtk.vtkActor() for i in range(n)]
    renderer = vtk.vtkRenderer()
    
    for i in range(n):
        mappers[i].SetInputData(polydatas[i])
        actors[i].SetMapper(mappers[i])
        actors[i].GetProperty().SetColor(*cols[i][:3])
        if wireframe:
            actors[i].GetProperty().SetRepresentationToWireframe()
        renderer.AddActor(actors[i])
        
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    render_window.Render()
    render_window_interactor.Start()


def plot_mesh(pts, simps, lib='plotly'):
    
    if lib == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                i=simps[:,0], j=simps[:,1], k=simps[:,2],
                                opacity=1, color='lightblue'))
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                   mode='markers', marker=dict(size=5, color='blue', opacity=1)))
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show(auto_open=True)
    
    elif lib == 'pyvista':
        simps_pv = np.hstack([[3, *tri] for tri in simps])
        mesh = pv.PolyData(pts, simps_pv)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="lightseagreen", show_edges=True)
        plotter.add_axes()
        plotter.show_grid()
        plotter.show()
