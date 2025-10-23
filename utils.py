import numpy as np
import matplotlib.pyplot as plt
import skimage
# import jax.numpy as jnp
from scipy.linalg import logm    # not in jax.scipy yet...
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import pyvista as pv


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
   
    

def seg_to_contour(seg, decim=1, get_simps=True, get_normals=False):

    opts_list = skimage.measure.find_contours(seg > 0)
    
    opts_list = [opts[:,[1,0]] for opts in opts_list]
    
    pts, simps, normals = opts_to_contour(opts_list, decim=decim, get_simps=get_simps, get_normals=get_normals)

    return pts, simps, normals


def pts_sqdist(pts1, pts2):
    
    diff = pts1[:,None,:] - pts2[None,:,:]
    
    return jnp.sum(diff ** 2, axis=-1)


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
                 col=[1,0,0], linewidth=1, markersize=3,
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
            
    plt.plot(pts[:,0], pts[:,1], '.', markersize, color=col_pts)      
    
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


def phase_align_contours(pts1, pts2, simps2=None):
    """
    assumes 2D pts, ordered and npts1 = npts2
    """

    npts = pts1.shape[0]
    best_k = 0
    best_dist = np.inf
    
    for k in range(npts):
        pts2_k = np.roll(pts2, shift=k, axis=0)
        dist = np.sum(np.linalg.norm(pts1 - pts2_k, axis=1))
        if dist < best_dist:
            best_dist = dist
            best_k = k
    
    pts2 = np.roll(pts2, shift=best_k, axis=0)
    
    if simps2 is None:
        return pts2
    else: 
        simps2 = (simps2 + best_k) % npts
        return pts2, simps2
    


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



def plot_mesh(pts, simps, lib='plotly'):
    
    if lib == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                i=simps[:,0], j=simps[:,1], k=simps[:,2],
                                opacity=1, color='lightblue'))
        # fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
        #                            mode='markers', marker=dict(size=5, color='blue', opacity=1)))
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
