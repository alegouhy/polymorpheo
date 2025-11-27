import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm, logm
import copy

import utils
import register



class io():
    
    def __init__(self, datadir, names, spacing, npts=None):
        
        self.spacing = spacing
        self.npts = npts
        self.nlabs = len(names)
        self.names = names
        self.files = [os.path.join(datadir, name + '.npz') for name in names]
        
    def load(self, plot=False):
        
        opts_lists = [np.load(file, allow_pickle=True)['registered_contours'] for file in self.files]
        nslices = len(opts_lists[0])
  
        pts_all = np.vstack([opt for opts_list in opts_lists for opts in opts_list if opts is not None for opt in opts])
        self.pts_mu = np.mean(pts_all, axis=0)
        self.pts_amp = np.max(np.abs(pts_all - self.pts_mu))
        
        polylines = []
        for i in range(nslices):
        
            polyline = []
            for l in range(self.nlabs):
                opts = opts_lists[l][i]
                if opts is None: continue
                polyline_l = utils.opts_to_contour(opts, npts=self.npts, get_simps=True, lab=l+1)
                polyline.append(polyline_l)
            if len(polyline) == 0: continue
        
            pts, simps, _, labs = utils.concat_contours(polyline)
            
            pts = (pts - self.pts_mu) / self.pts_amp
            
            polyline = pts, simps, None, labs
            polylines.append(polyline)
            
            if plot:
                utils.plot_contour(polyline)
                plt.title(str(i))
                plt.show()
          
        return polylines
    
    
    def save(self, meshes, outdir, suffix):
        
        if suffix != '': suffix = '_' + suffix
        
        for l in range(self.nlabs):
            
            pts, simps = meshes[l]
            pts[:,:2] = (pts[:,:2] * self.pts_amp + self.pts_mu) * self.spacing[:2]
            pts[:,2] = pts[:,2] * self.spacing[2]
            
            poly = utils.vtkpoly(pts, simps)
            out_file = os.path.join(outdir, self.names[l] + suffix + '.obj')
            utils.write_vtkpoly(poly, out_file)
      
    
    
    
class register_slices():
    
    def __init__(self, method, transfo, init, niter=1, degree=2, icp_niter=30, bidir=True, plot=False):
        
        self.method = method   
        self.niter = niter
        if transfo in ('rig', 'rigid', 'aff', 'affine'):
            self.transfo_type = 'linear'
            self.reg = register.reg_linear(niter=icp_niter, transfo=transfo, init=init, se=True, bidir=bidir, plot=plot)
        elif transfo in ('poly', 'polynom'):
            self.transfo_type = 'polynom'
            self.reg = register.reg_polynom(niter=icp_niter, degree=degree, init=init, se=True, bidir=bidir, plot=plot)
            
  
    def compute(self, polylines):
        
        if self.method == 0:
            return self._method_0(polylines)
        elif self.method == 1:
            return self._method_1(polylines)
        elif self.method == 2:
            return self._method_2(polylines)
        elif self.method == 3:
            return self._method_3(polylines)
        elif self.method == 4:
            return self._method_4(polylines)
        elif self.method == 5:
            return self._method_5(polylines)
        
        
    def _method_0(self, polylines):
        
        nslice = len(polylines)
        midslice = int(nslice / 2)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        
        for _ in range(self.niter):
            
            for i in range(midslice,nslice-1):
                
                ref_polyline = polylines[i]
                mov_polyline = polylines0[i+1]
                
                if self.transfo_type == 'linear':
                    _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)     
                elif self.transfo_type == 'polynom':
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    
                polylines[i+1] = moved_polyline
               
            for i in reversed(range(1,midslice)):
                
                ref_polyline = polylines[i]
                mov_polyline = polylines[i-1]
    
                if self.transfo_type == 'linear':
                    _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    
                elif self.transfo_type == 'polynom':
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                    
                polylines[i-1] = moved_polyline
            
            polylines0 = polylines.copy()
            
        return polylines
            
        
    def _method_1(self, polylines):
        
        nslice = len(polylines)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)

        for _ in range(self.niter):

            for i in range(1, nslice-1):
                
                prev_polyline = polylines[i-1]
                next_polyline = polylines0[i+1]
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                
                if self.transfo_type == 'linear':
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    
                elif self.transfo_type == 'polynom':
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2  
                    
                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline
            
            polylines0 = polylines.copy()
            
        return polylines
        
    
    def _method_2(self, polylines):
        
        nslice = len(polylines)
        midslice = int(nslice / 2)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        
        for _ in range(self.niter):
    
            for i in range(midslice,nslice-1):

                prev_polyline = polylines[i-1]
                next_polyline = polylines0[i+1]
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
    
                if self.transfo_type == 'linear':
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    
                elif self.transfo_type == 'polynom':
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2  
                    
                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline  
                
            for i in reversed(range(1,midslice)):

                prev_polyline = polylines[i+1]
                next_polyline = polylines0[i-1]
                mov_polyline = polylines0[i]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                
                if self.transfo_type == 'linear':
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans
                    
                elif self.transfo_type == 'polynom':
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2  
                    
                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline

            polylines0 = polylines.copy()
            
        return polylines
    
    
    def _method_3(self, polylines):
        
        nslice = len(polylines)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)

        for _ in range(self.niter):
        
            for i in range(1, nslice, 2):
                
                mov_polyline = polylines0[i]
                prev_polyline = polylines0[i-1]
                next_polyline = polylines0[i+1] if i < nslice - 1 else None
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                
                if self.transfo_type == 'linear':
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else (np.eye(3), None)             
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans                 
                    
                elif self.transfo_type == 'polynom':
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline)
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline) if i < nslice - 1 else mov_polyline 
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                
                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline
             
            polylines_aff0 = polylines.copy()
            
            for i in range(0, nslice, 2):
                
                mov_polyline = polylines0[i]
                prev_polyline = polylines_aff0[i-1] if i > 0 else None
                next_polyline = polylines_aff0[i+1]
                mov_pts, mov_simps, _, mov_labs = mov_polyline
                
                if self.transfo_type == 'linear':
                    aff_prev, _ = self.reg.compute(prev_polyline, mov_polyline) if i > 0 else (np.eye(3), None)
                    aff_next, _ = self.reg.compute(next_polyline, mov_polyline)    
                    aff = np.real(expm((logm(aff_prev, disp=False)[0] + logm(aff_next, disp=False)[0]) / 2))
                    lin, trans = utils.aff_dehmgn(aff)
                    moved_pts = mov_pts @ lin.T + trans                 
                    
                elif self.transfo_type == 'polynom':
                    moved_polyline_prev = self.reg.compute(prev_polyline, mov_polyline) if i > 0 else mov_polyline 
                    moved_polyline_next = self.reg.compute(next_polyline, mov_polyline)
                    moved_pts_prev, _, _, _ = moved_polyline_prev
                    moved_pts_next, _, _, _ = moved_polyline_next
                    moved_pts = (moved_pts_prev + moved_pts_next) / 2
                
                moved_polyline = moved_pts, mov_simps, None, mov_labs
                polylines[i] = moved_polyline
             
            polylines0 = polylines.copy()
            
        return polylines
        
    
    def _method_4(self, polylines):
        
        nslice = len(polylines)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)

        for _ in range(self.niter):

            for i in range(1, nslice, 2):
                
                mov_polyline = polylines0[i]
                ref_polyline = [polylines0[i-1]]
                if i < nslice - 1: ref_polyline += [polylines0[i+1]]   
                ref_polyline = utils.concat_contours(ref_polyline)
                
                if self.transfo_type == 'linear':
                    _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == 'polynom':
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
    
                polylines[i] = moved_polyline
             
            polylines0 = polylines.copy()
            
            for i in range(0, nslice, 2):
                
                mov_polyline = polylines0[i]   
                ref_polyline = [polylines0[i+1]]
                if i > 0: ref_polyline += [polylines0[i-1]]
                ref_polyline = utils.concat_contours(ref_polyline)
                
                if self.transfo_type == 'linear':
                    _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                elif self.transfo_type == 'polynom':
                    moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
    
                polylines[i] = moved_polyline
             
            polylines0 = polylines.copy()
            
        return polylines
    
    
    def _method_5(self, polylines):
        
        nslice = len(polylines)
        midslice = int(nslice / 2)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        
        for _ in range(self.niter):
            
            for i in range(midslice+1, nslice-1):               
                # plt.subplot(3,3,1); utils.plot_contour(polylines0[i-1])
                # plt.subplot(3,3,2); utils.plot_contour(polylines0[i])
                # plt.subplot(3,3,3); utils.plot_contour(polylines0[i+1])
                # plt.subplot(3,3,4); utils.plot_contour(polylines[i-1])
                # plt.subplot(3,3,5); utils.plot_contour(polylines[i])
                # plt.subplot(3,3,6); utils.plot_contour(polylines[i+1])
                
                mov_polyline = polylines0[i]
                ref_polyline = [polylines[i-1], polylines0[i+1]]
                ref_polyline = utils.concat_contours(ref_polyline)
                _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                polylines[i] = moved_polyline
            
                # plt.subplot(3,3,7); utils.plot_contour(polylines[i-1])
                # plt.subplot(3,3,8); utils.plot_contour(polylines[i])
                # plt.subplot(3,3,9); utils.plot_contour(polylines[i+1])
                # plt.suptitle(i); plt.show()
            
            polylines0 = copy.deepcopy(polylines)
                
            for i in reversed(range(1, midslice)):
                # plt.subplot(3,3,1); utils.plot_contour(polylines0[i-1])
                # plt.subplot(3,3,2); utils.plot_contour(polylines0[i])
                # plt.subplot(3,3,3); utils.plot_contour(polylines0[i+1])
                # plt.subplot(3,3,4); utils.plot_contour(polylines[i-1])
                # plt.subplot(3,3,5); utils.plot_contour(polylines[i])
                # plt.subplot(3,3,6); utils.plot_contour(polylines[i+1])
                
                mov_polyline = polylines0[i]
                ref_polyline = [polylines0[i-1], polylines[i+1]]
                ref_polyline = utils.concat_contours(ref_polyline)
                _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                polylines[i] = moved_polyline
                
                # plt.subplot(3,3,7); utils.plot_contour(polylines[i-1])
                # plt.subplot(3,3,8); utils.plot_contour(polylines[i])
                # plt.subplot(3,3,9); utils.plot_contour(polylines[i+1])
                # plt.suptitle(i); plt.show()
            
            polylines0 = copy.deepcopy(polylines)
            
        return polylines
    
            
class bridge_contours():
    
    def __init__(self, thr_conn=1/3, greedy=False):
        
        self.thr_conn = thr_conn
        self.greedy = greedy
        
        
    def compute(self, polylines):       
        
        nslices = len(polylines)
        labs = np.unique(np.concatenate([polyline[3] for polyline in polylines]))
        z_coords = np.arange(nslices)
        
        meshes = []
        for l in range(len(labs)):
            
            polylines_l = [utils.extract_polyline(polyline, polyline[3] == l + 1) for polyline in polylines]
            opts_list = []
            
            for i in range(nslices):
                
                polyline_l = polylines_l[i]
                opts = utils.contours2opts(np.array(polyline_l[0]), np.array(polyline_l[1]))
                opts_list.append(opts)
                
            pts, simps = utils.bridge_contours_2(opts_list, z_coords, greedy=self.greedy, thr_conn=self.thr_conn)
            meshes.append([pts, simps])
        
        return meshes
    
    
    
    def _method_5(self, polylines):
        
        nslice = len(polylines)
        midslice = int(nslice / 2)
        polylines = copy.deepcopy(polylines)
        polylines0 = copy.deepcopy(polylines)
        
        for _ in range(self.niter):
            
            for i in range(midslice+1, nslice-1):               

                mov_polyline = polylines0[i]
                ref_polyline = [polylines[i-1], polylines0[i+1]]
                ref_polyline = utils.concat_contours(ref_polyline)
                _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                polylines[i] = moved_polyline
            
            polylines0 = copy.deepcopy(polylines)
                
            for i in reversed(range(1, midslice)):

                mov_polyline = polylines0[i]
                ref_polyline = [polylines0[i-1], polylines[i+1]]
                ref_polyline = utils.concat_contours(ref_polyline)
                _, moved_polyline = self.reg.compute(ref_polyline, mov_polyline)
                polylines[i] = moved_polyline

            polylines0 = copy.deepcopy(polylines)
            
        return polylines