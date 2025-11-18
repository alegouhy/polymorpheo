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
from shapely.geometry import LinearRing, Polygon, Point
from shapely import make_valid

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
   

def close_contour(opt):
    
    if np.all(opt[0,:] != opt[-1,:]):
        np.vstack((opt, opt[0,:]))
        
    return opt

def open_contour(opt):
    
    while np.all(opt[0,:] == opt[-1,:]) and opt.shape[0] > 1:
        opt = opt[:-1,:]

    return opt


def split_opts(opts, i, j):
    # with i < j
    
    opts1 = np.vstack((opts[:i+1, :], opts[j:, :]))
    opts2 = np.vstack((opts[i:j+1, :]))
    
    return [opts1, opts2]


def split_opts_2(opts, i, j):
    # with i < j
    
    npts = opts.shape[0]
    dist = np.sum(np.linalg.norm(np.diff(opts, axis=0), axis=1))
    ro = npts / dist
    
    if i == j:
        opts1 = opts.copy()
        opts2 = opts[i,:][None,...]
        
    else:
        dist_link = np.linalg.norm(opts[i,:]-opts[j,:])
        npts_link = np.max((np.round(ro * dist_link).astype(int), 2))
        link = np.linspace(opts[i,:], opts[j,:], npts_link)
        
        opts1 = np.vstack((opts[:i, :], link, opts[j+1:, :]))
        opts2 = np.vstack((opts[i+1:j, :], np.flip(link, axis=0)))
    
    return [opts1, opts2]



# def splitfit_opts(opts_1, opts_2):

#     opts_1 = [open_contour(opt) for opt in opts_1]
#     opts_2 = [open_contour(opt) for opt in opts_2]
    
#     pt_2_rep = [representative_point(opt) for opt in opts_2]
                
#     dist_6 = chamfer(opts_1[0], opts_2[0]) + \
#              chamfer(pt_2_rep[1], opts_2[1]) 
#     dist_7 = chamfer(opts_1[0], opts_2[1]) + \
#              chamfer(pt_2_rep[0], opts_2[0]) 
    
#     ind_dist_hat = np.argmin([dist_6, dist_7])
    
#     if ind_dist_hat == 0:
#         opts_1_split = [opts_1[0], pt_2_rep[1]]
#     elif ind_dist_hat == 0:
#         opts_1_split = [opts_1[0], pt_2_rep[0]]
                        
#     opts_1_split = [open_contour(opt) for opt in opts_1_split]
    
#     plt.figure()
#     plt.subplot(2,2,1)
#     plot_opts(opts_1_split[0], markersize=1)
#     plt.subplot(2,2,2)
#     plot_opts(opts_1_split[1], markersize=1)
#     plt.subplot(2,2,3)
#     plot_opts(opts_2[0], markersize=1)
#     plt.subplot(2,2,4)
#     plot_opts(opts_2[1], markersize=1)
#     plt.show()
    
#     opts_1_split = cw(opts_1_split)
    
#     plt.figure()
#     plt.subplot(2,2,1)
#     plot_opts(opts_1_split[0])
#     plt.subplot(2,2,2)
#     plot_opts(opts_1_split[1])
#     plt.subplot(2,2,3)
#     plot_opts(opts_2[0])
#     plt.subplot(2,2,4)
#     plot_opts(opts_2[1])
#     plt.show()

#     return opts_1_split



def splitfit_opts(opts_1, opts_2):

    opts_1 = [open_contour(opt) for opt in opts_1]
    opts_2 = [open_contour(opt) for opt in opts_2]
    n = opts_1[0].shape[0]
    
    i_hat = 0
    j_hat = 0
    dist_hat = np.inf
    
    for opt in opts_1:
        plot_opts(opt, col=[1,0,0])
        plt.show()
    for opt in opts_2:
        plot_opts(opt, col=[0,0,1])
        plt.show()
    
    for i in range(1, n):
        for j in range(i, n):
            
            opts_1_split = split_opts_2(opts_1[0], i, j)
            opts_1_split = [open_contour(opt) for opt in opts_1_split]
            
            dist0 = chamfer(opts_1_split[0], opts_2[0]) + \
                    chamfer(opts_1_split[1], opts_2[1])

            dist1 = chamfer(opts_1[0], opts_2[0]) + \
                    chamfer(opts_1_split[1], opts_2[1])
                    
            dist2 = chamfer(opts_1[0], opts_2[0]) + \
                    chamfer(opts_1_split[0], opts_2[1])
                    
            dist3 = chamfer(opts_1_split[0], opts_2[1]) + \
                    chamfer(opts_1_split[1], opts_2[0])
                    
            dist4 = chamfer(opts_1[0], opts_2[1]) + \
                    chamfer(opts_1_split[1], opts_2[0])
                    
            dist5 = chamfer(opts_1[0], opts_2[1]) + \
                    chamfer(opts_1_split[0], opts_2[0])
                    
            dists = [dist0, dist1, dist2, dist3, dist4, dist5]
            ind_dist = np.argmin(dists)
            dist = dists[ind_dist]
            
            if dist < dist_hat:
                i_hat, j_hat = i, j
                dist_hat = dist.copy()
                ind_dist_hat = ind_dist.copy()
    
    opts_1_split = split_opts_2(opts_1[0], i_hat, j_hat)
    opts_1_split = [open_contour(opt) for opt in opts_1_split]
    
    if ind_dist_hat in (1,4): 
        opts_1_split = [opts_1[0], opts_1_split[1]]
    elif ind_dist_hat in (2,5): 
        opts_1_split = [opts_1[0], opts_1_split[0]]
    
    if ind_dist_hat > 2:
        opts_1_split = [opts_1_split[1], opts_1_split[0]]
        
    # plt.figure()
    # plt.subplot(2,2,1)
    # plot_opts(opts_1_split[0], markersize=1)
    # plt.subplot(2,2,2)
    # plot_opts(opts_1_split[1], markersize=1)
    # plt.subplot(2,2,3)
    # plot_opts(opts_2[0], markersize=1)
    # plt.subplot(2,2,4)
    # plot_opts(opts_2[1], markersize=1)
    # plt.show()
    
    opts_1_split = cw(opts_1_split)
    
    # plt.figure()
    # plt.subplot(2,2,1)
    # plot_opts(opts_1_split[0])
    # plt.subplot(2,2,2)
    # plot_opts(opts_1_split[1])
    # plt.subplot(2,2,3)
    # plot_opts(opts_2[0])
    # plt.subplot(2,2,4)
    # plot_opts(opts_2[1])
    # plt.show()

    return opts_1_split


# def splitfit_opts(opts_1, opts_2):

#     i_hat = 0
#     j_hat = 0
#     dist_hat = np.inf
    
#     n = opts_1[0].shape[0]
    
#     for i in range(n):
#         for j in range(n):
#             if i > j: continue
            
#             opts_1_split = split_opts_2(opts_1[0], i, j)
            
#             dist1 = chamfer(opts_1_split[0], opts_2[0]) + \
#                     chamfer(opts_1_split[1], opts_2[1])
            
#             dist2 = chamfer(opts_1_split[0], opts_2[1]) + \
#                     chamfer(opts_1_split[1], opts_2[0])
            
#             dist = min(dist1, dist2)
            
#             if dist < dist_hat:
#                 i_hat = i
#                 j_hat = j
#                 dist_hat = dist
#                 swap = True if dist2 < dist1 else False
    
#     opts_1_split = split_opts_2(opts_1[0], i_hat, j_hat)
#     opts_1_split = cw(opts_1_split)
#     if swap:
#         opts_1_split = [opts_1_split[1], opts_1_split[0]]
    
#     cols = [[0,0,1],[1,0,0]]
#     plt.figure()
#     for i, opt in enumerate(opts_1_split):
#         plot_opts(opt, col=cols[i])
#     plt.show()
    
#     plt.figure()
#     for i, opt in enumerate(opts_1_split):
#         plot_opts(opt, col=cols[i])
#         plt.show()
    
#     return opts_1_split


def seg_to_contour(seg, npts=None, get_simps=True, get_normals=False):

    opts_list = skimage.measure.find_contours(seg > 0)
    
    opts_list = [opts[:,[1,0]] for opts in opts_list]
    
    pts, simps, normals = opts_to_contour(opts_list, npts=npts, get_simps=get_simps, get_normals=get_normals)

    return pts, simps, normals


@jax.jit
def pts_sqdist(pts1, pts2):
    
    diff = pts1[:,None,:] - pts2[None,:,:]
    
    return jnp.sum(diff ** 2, axis=-1)

@jax.jit
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
    for simp in simps:
    
        adj[simp[0], simp[1]] = 1
        adj[simp[1], simp[0]] = 1
        if ndims == 3:
            adj[simp[1], simp[2]] = 1
            adj[simp[2], simp[1]] = 1
            adj[simp[2], simp[0]] = 1
            adj[simp[0], simp[2]] = 1

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


def concat_contours(contours, z_coords=None):
    
    is_simps = contours[0][1] is not None
    is_normals = contours[0][2] is not None
    
    pts_all = []
    simps_all = [] if is_simps else None
    normals_all = [] if is_normals else None
    offset = 0
    for c, contour in enumerate(contours):
        
        pts, simps, normals = contour
        
        npts = pts.shape[0]
        if z_coords is not None:
            pts = np.concatenate((pts,np.full((npts,1), z_coords[c])), axis=1)
        
        pts_all.append(pts)
        if is_simps:
            simps_all.append(simps + offset)
        if is_normals:
            normals_all.append(normals)
            
        offset += npts
        
    pts_all = np.concatenate(pts_all, axis=0)
    if is_simps:
        simps_all = np.concatenate(simps_all, axis=0)
    if is_normals:
        normals_all = np.concatenate(normals_all, axis=0)
        
    return pts_all, simps_all, normals_all


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


def representative_point(opts):
    
    opts_c = close_contour(opts)
    if opts_c.shape[0] < 4:
        pt_rep = np.mean(opts, axis=0)
    else:
        pt_rep = Polygon(opts_c).representative_point()
        pt_rep = np.array((pt_rep.x, pt_rep.y))
        
    return pt_rep[None,...] 
    

def conn_events(nodes_curr, nodes_next, links):

    out_map = {a: [] for a in nodes_curr}
    in_map  = {b: [] for b in nodes_next}

    for a, b in links:
        out_map[a].append(b)
        in_map[b].append(a)

    events = {"continue": [],
              "split": [], "merge": [],
              "birth": [], "death": []}
    
    merges_added = set()
    
    # death, continue or split
    for a in nodes_curr:
        outs = out_map[a]
        if len(outs) == 0:
            events["death"].append(a)
        elif len(outs) == 1:
            b = outs[0]
            if len(in_map[b]) == 1:
                events["continue"].append((a, b))
            else:
                # Only add merge once per target b
                if b not in merges_added:
                    events["merge"].append((in_map[b], b))
                    merges_added.add(b)
        else:
            events["split"].append((a, outs))

    # birth
    for b in nodes_next:
        if len(in_map[b]) == 0:
            events["birth"].append(b)

    return events



def bridge_contours_2(opts_list, z_coords, thr_conn=1/3, greedy=False, sealed=True):
    
    pts = []
    simps = []
    offset = 0
    if greedy: 
        path_fun = triangle_path_greedy
    else:
        path_fun = triangle_path_dp
    
    nodes_list, links_list = build_graph_conn(opts_list, thr_conn)
    plot_graph_conn(nodes_list, links_list)
    plt.show()
    
    if sealed:
        lid_start = [representative_point(opts) for opts in opts_list[0]]
        lid_end = [representative_point(opts) for opts in opts_list[-1]]
        opts_list = [lid_start] + opts_list + [lid_end]
        z_coords = np.hstack((z_coords[0]-z_coords[1], z_coords, z_coords[-1]))
        nodes_list = [nodes_list[0]] + nodes_list + [nodes_list[-1]]
        links_list = [[[node, node] for node in nodes_list[0]]] + links_list + [[[node, node] for node in nodes_list[-1]]]
    
    for i in range(len(opts_list)-1):
        print(i)
        
        opts_1 = cw(opts_list[i])
        opts_2 = cw(opts_list[i+1])  
        links = links_list[i]
        nodes_1 = nodes_list[i]
        nodes_2 = nodes_list[i+1]
        
        events = conn_events(nodes_1, nodes_2, links)
        
        opts_1_proc = []
        opts_2_proc = []
        
        for j_curr, js_next in events['split']:
            parts_next = [opts_2[j] for j in js_next]
            parts_curr = splitfit_opts([opts_1[j_curr]], parts_next)
            opts_1_proc += parts_curr
            opts_2_proc += parts_next
            
        for js_curr, j_next in events['merge']:
            parts_curr = [opts_1[j] for j in js_curr]
            parts_next = splitfit_opts([opts_2[j_next]], parts_curr)
            opts_1_proc += parts_curr
            opts_2_proc += parts_next
            
        for j_curr in events['death']:
            part_curr = opts_1[j_curr]
            pt_rep_curr = representative_point(part_curr)
            opts_1_proc += [part_curr]
            opts_2_proc += [pt_rep_curr]
        
        for j_next in events['birth']:
            part_next = opts_2[j_next]
            pt_rep_next = representative_point(part_next)
            opts_1_proc += [pt_rep_next]
            opts_2_proc += [part_next]

        for j_curr, j_next in events['continue']:
            opts_1_proc += [opts_1[j_curr]]
            opts_2_proc += [opts_2[j_next]]
            
        n = len(opts_1_proc)

        for k in range(n):
            
            opts_1k = open_contour(opts_1_proc[k])
            opts_2k = open_contour(opts_2_proc[k])
            nn1 = opts_1k.shape[0]
            nn2 = opts_2k.shape[0]
            
            opts_2k = phase_align_contours(opts_1k, opts_2k)
            
            # plt.subplot(2,n1,k+1)
            plt.figure()
            plot_opts(opts_1k, col=[1,0,0])
            plot_opts(opts_2k, col=[0,0,1])
            plt.title(str(i) + ', ' + str(k))
            plt.show()
            
            opts_1k = np.hstack([opts_1k, np.full((opts_1k.shape[0],1), z_coords[i])])
            opts_2k = np.hstack([opts_2k, np.full((opts_2k.shape[0],1), z_coords[i+1])])
            pts_k = np.vstack([opts_1k, opts_2k])
             
            path, path_len = path_fun(opts_1k, opts_2k)
            _, simps_k = triangulate_path(path, opts_1k, opts_2k)
            
            plt.subplot(2, n, n+k+1)
            plot_path(path, nn1, nn2)
            
            simps_k = simps_k + offset
            
            pts.append(pts_k)
            simps.append(simps_k)
            offset += pts_k.shape[0]

        plt.suptitle(str(i))
        plt.show()
                 
    pts = np.vstack(pts)
    simps = np.vstack(simps)

    pts, simps = clean_mesh(pts, simps)

    return pts, simps


    
def rm_dup_pts(pts, simps):

    pts_u, inv = np.unique(pts, axis=0, return_inverse=True)
    simps_u = inv[simps]
    
    return pts_u, simps_u

def rm_dup_simps(simps):
    
    simps_u = np.sort(simps, axis=1)
    _, ind = np.unique(simps_u, axis=0, return_index=True)
    simps_u = simps[ind, :]
    
    return simps_u

def rm_degen_simps(simps):
    
    ndims = simps.shape[1]
    
    valid = (simps[:, 0] != simps[:, 1])
    if ndims > 2:
        valid = valid & (simps[:, 1] != simps[:, 2]) \
                      & (simps[:, 2] != simps[:, 0])

    return simps[valid]

def rm_single_pts(pts, simps):
    
    ind_pts = np.unique(simps)
    
    old_to_new = np.full(len(pts), np.nan)
    old_to_new[ind_pts] = np.arange(len(ind_pts))
    
    pts_clean = pts[ind_pts]
    simps_clean = old_to_new[simps].astype(simps.dtype)
    
    return pts_clean, simps_clean

    
    
def clean_mesh(pts, simps):
    
    pts_clean, simps_clean = rm_dup_pts(pts, simps)
    simps_clean = rm_dup_simps(simps_clean)
    simps_clean = rm_degen_simps(simps_clean)
    pts_clean, simps_clean = rm_single_pts(pts_clean, simps_clean)
    
    return pts_clean, simps_clean

    

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



def triangulate_path(path, pts1, pts2):
    
    npts1 = pts1.shape[0]
    npts2 = pts2.shape[0]
    simps = []
    
    for k in range(len(path) - 1):
        
        i_curr, j_curr = int(path[k, 0]), int(path[k, 1])
        i_next, j_next = int(path[k + 1, 0]), int(path[k + 1, 1])
        
        di = i_next - i_curr
        dj = j_next - j_curr
        
        if di == 1 and dj == 0:
            simps.append([i_curr, npts1 + j_curr, i_next])
            
        elif di == 0 and dj == 1:
            simps.append([i_curr, npts1 + j_curr, npts1 + j_next])
            
        elif di == 1 and dj == 1:
            simps.append([i_curr, npts1 + j_curr, i_next])
            simps.append([i_next, npts1 + j_curr, npts1 + j_next])
            
    simps.append([npts1 - 1, npts1 + npts2 - 1, 0])
    simps.append([0, npts1 + npts2 - 1, npts1])

    return jnp.vstack((pts1, pts2)), jnp.array(simps)



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
    
    dsimp = simps.shape[1]
    
    if lib == 'plotly':
        fig = go.Figure()
        if dsimp ==3:
            fig.add_trace(go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                    i=simps[:,0], j=simps[:,1], k=simps[:,2],
                                    opacity=1, color='lightblue'))
        elif dsimp == 2:
            fig.add_trace(go.Mesh3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                    i=simps[:,0], j=simps[:,1],
                                    opacity=1, color='lightblue'))
            
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                   mode='markers', marker=dict(size=5, color='blue', opacity=1)))
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.show(auto_open=True)
    
    elif lib == 'pyvista':
        simps_pv = np.hstack([[dsimp, *tri] for tri in simps])
        mesh = pv.PolyData(pts, simps_pv)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="lightseagreen", show_edges=True)
        plotter.add_axes()
        plotter.show_grid()
        plotter.show()


def graph_conv(A, X, K, normA='left', niter=1):
    # normA: 'left' or 'right' or 'sym' or 'sinkhorn'
    
    A = A + np.eye(A.shape[0], dtype=int)
    
    if normA=='sinkhorn':
        A_norm = A
        for i in range(1000):
            deg = np.sum(A_norm, axis=1)
            D = np.diag(1/deg)
            A_norm = np.matmul(D,A_norm)
            deg = np.sum(A_norm, axis=0)
            D = np.diag(1/deg)
            A_norm = np.matmul(A_norm,D)
    else:        
        deg = np.sum(A, axis=0)
        if normA == 'sym':
            D = np.diag(1/np.sqrt(deg))
            A_norm = np.matmul(np.matmul(D,A),D)
        else:
            D = np.diag(1/deg)
            if normA == 'left':
                A_norm = np.matmul(D,A)
            elif normA == 'right':
                A_norm = np.matmul(A,D)
    
    for _ in range(niter):
        X = np.matmul(np.matmul(A_norm, X), K)
    
    return X


def build_graph_conn(opts_list, thr):
    
    links_list = []
    nodes_list = []
    for i in range(len(opts_list) - 1):
        
        opts_curr = cw(opts_list[i])
        opts_next = cw(opts_list[i+1])
        n_curr = len(opts_curr)
        n_next = len(opts_next)
    
        links = []
        nodes = []
        for j in range(n_curr):
            opt_curr = open_contour(opts_curr[j])
            
            if opt_curr.shape[0] > 2:
                poly_curr = Polygon(opt_curr)
                poly_curr = make_valid(poly_curr)
            else:
                poly_curr = [Point(pt) for pt in opt_curr]
                
            for k in range(n_next):
                opt_next = open_contour(opts_next[k])
     
                if opt_next.shape[0] > 2:
                    poly_next = Polygon(opt_next)
                    poly_next = make_valid(poly_next)
                else:
                    poly_next = [Point(pt) for pt in opt_next]
                    
                if opt_curr.shape[0] > 2 and opt_next.shape[0] > 2:
                    inter = poly_curr.intersection(poly_next)
                    overlap_1 = inter.area / poly_curr.area
                    overlap_2 = inter.area / poly_next.area
                    linked =  np.max((overlap_1, overlap_2)) > thr
                    
                elif opt_curr.shape[0] > 2 and opt_next.shape[0] < 3:
                    linked = any([pt.within(poly_curr) for pt in poly_next])
                    
                elif opt_curr.shape[0] < 3 and opt_next.shape[0] > 2:
                    linked = any([pt.within(poly_next) for pt in poly_curr])
                
                else: linked = False
                        
                if linked:
                    links.append([j, k])
                    
            nodes.append(j)
        nodes_list.append(nodes)
        
        if i == len(opts_list) - 2:
            nodes = []
            for j in range(n_next):
                nodes.append(j)
            nodes_list.append(nodes)
        
        links_list.append(links)   

    return nodes_list, links_list


def plot_graph_conn(nodes_list, links_list, col=[0,0,1]):
    
    n = len(nodes_list)
    col_nodes = np.array(col)
    col_links = (1 + col_nodes) / 2
    plt.figure(figsize=(6,1))
    
    for i in range(n):
        if i < n - 1:
            links = links_list[i]
            for j in range(len(links)):
                plt.plot([i, i+1], links[j], color=col_links)
        
        nodes = nodes_list[i]
        for j in range(len(nodes)):
            plt.plot(i, nodes[j], '.', markersize=3, color=col_nodes)
            
    plt.ylim([-0.5, 1.5])  
    plt.xlim([-1, n])
    plt.show()
