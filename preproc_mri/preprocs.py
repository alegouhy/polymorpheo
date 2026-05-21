import os
import sys
import glob
import nibabel as nib
import numpy as np

#%%

RAW_DIR = '/Users/alegouhy/data/ferret-mri/data/raw/'
SRC_DIR = '/Users/alegouhy/data/ferret-mri/src'
sys.path.append('SRC_DIR')
from plot_anat import plot_anat


#%% step 01 - t2 - diagnosis

DERIMA_DIR = '/Users/alegouhy/tests/atlas_ferret'

sub = '15'
tpt = '16'
seq = 'MSME'
acq =  'F' + sub + '_P' + tpt

img_file = glob.glob(os.path.join(RAW_DIR, acq, 't2', '*', 'anatomy', '*' + seq + '*TEsum_magn.nii.gz'))[0]
_, img_name = os.path.split(img_file)
img = nib.load(img_file)
out_file = os.path.join(DERIMA_DIR, 'step-01_' + img_name[:-7] + '.png')
plot_anat(img, output_file=out_file)


#%% step 02 - t2 - fix rotation rename

DERCON_DIR = '../data/derived/converted/'

SUB = [['F01_Adult', 0, 'Adult_F01_31102012/msme_8/anatomy/F01_31102012_MSME_TEsum_magn', 'msme_1',  1.0],    
       ['F15_P16',   3, 'P16_F15_07022014/anatomy/P16_F15_07022014_MSME_TEsum_magn',      'msme_1', 10.0]]
    
def scale_rotate(src, dst, sca, rot):
    """Scale and rotate a nifti volume"""
    img = nib.load(src)
    print("orig:" + str(np.linalg.det(img.affine)))
    if (sca != 1) and (sca != 0):
        (img.affine[:3, :])[:] = (img.affine[:3, :])[:]/sca
    img.affine[:] = rot.dot(img.affine)
    print("new:" + str(np.linalg.det(img.affine)))
    nib.nifti1.save(img, dst)

def main(argv):
    """Scale and rotate nifti volumes for all subjects"""
    rot0 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # OK
    rot1 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # OK
    rot2 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # OK
    rot3 = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # OK
    rot4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) # OK

    print("rot0: " + str(np.linalg.det(rot0)))
    print("rot1: " + str(np.linalg.det(rot1)))
    print("rot2: " + str(np.linalg.det(rot2)))
    print("rot3: " + str(np.linalg.det(rot3)))
    print("rot4: " + str(np.linalg.det(rot4)))

    arr = {}
    for row in SUB:
        sub, group, src, dst, sca = row

        if group == 0:
            rot = rot0
        elif group == 1:
            rot = rot1
        elif group == 2:
            rot = rot2
        elif group == 3:
            rot = rot3
        elif group == 4:
            rot = rot4

        try:
            # test overwrite
            nii_path = DERCON_DIR + sub + '/t2/' + dst + '.nii.gz'
            if OVERWRITE is False and os.path.lexists(nii_path):
                print("Skipping " + nii_path)
                continue
            print("Doing " + nii_path)
            # create destination directory if it doesn't exist
            if not os.path.lexists(DERCON_DIR + sub + '/t2/'):
                os.makedirs(DERCON_DIR + sub + '/t2/')
            # scale and rotate
            scale_rotate(RAW_DIR + sub + '/t2/' + src + '.nii.gz',
                         nii_path, sca, rot)
            print(src)
            if sub not in arr:
                arr[sub] = []
            arr[sub].append({
                'name': dst + '.nii.gz',
                'reference': RAW_DIR.replace('../', '') + sub + '/t2/' + src + '.nii.gz'
            })
        except:
            raise

    for sub in arr:
        obj = {}
        obj['references'] = arr[sub]
        print(json.dumps(obj, indent=2))
        print(DERCON_DIR + sub + '/t2/t2.json')
        with open(DERCON_DIR + sub + '/t2/t2.json', 'w') as file:
            json.dump(obj, file)

if __name__ == "__main__":
    main(sys.argv)
    
    
    
OVERWRITE = False

# load subject list
file = open("subjects.txt", "r")
SUB = [row.split(" ")[0] for row in file.read().split("\n") if len(row)>3]
file.close()

def process_one(sub):
    for root, _, files in os.walk(RAW_DIR + sub):
        for file in files:
            if not file.startswith('.') and file.endswith('TEsum_magn.nii.gz'):
                path = root + '/' + file
                ref = path.split('/')
                filename = ref[-2] + '-' + ref[-1].replace('.nii.gz', '') + '.png'
                dir = ref[3] + '/'

                if OVERWRITE == False and os.path.lexists(DERIMA_DIR + dir + filename):
                    print("Skipping %s%s%s"%(DERIMA_DIR,dir,filename))
                    continue
                print("Doing %s%s%s"%(DERIMA_DIR,dir,filename))
                try:
                    img = nib.load(path)

                    # create destination directory if it doesn't exist
                    if not os.path.exists(DERIMA_DIR + dir):
                        os.makedirs(DERIMA_DIR + dir)

                    # plot and save
                    plot_anat(img, output_file=DERIMA_DIR + dir + filename)
                except IOError:
                    raise
                except Exception as ex:
                    raise
                    
                    
                    