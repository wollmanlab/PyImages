# Metadata imports
from os import walk, listdir, path
from os.path import join, isdir
import sys
import tifffile
import cv2
import pandas
import numpy
import numpy as np
import os
from ast import literal_eval
import dill as pickle

from skimage import img_as_float, img_as_uint, io

#stkshow imports

class Metadata(object):
    def __init__(self, pth, md_name='Metadata.txt', load_type='local',low_memory=False,try_shortcut=True):
        """
        Load metadata files.
        """
        self.base_pth = pth
        self.low_memory = low_memory
        # short circuit recursive search for metadatas if present in the top directory of 
        # """ VERY SLOW WITH LARGE DATASETS """
        # the supplied pth.
        # if md_name in listdir(pth):
        #     self.image_table = self.load_metadata(join(pth), fname=md_name)
        # # Recursively append all metadatas in children directories of pth
        # else:
        #     all_mds = []
        #     for subdir, curdir, filez in walk(pth):
        #         if md_name in filez:
        #             try:
        #                 all_mds.append(self.load_metadata(join(pth, subdir), fname=md_name))
        #             except:
        #                 continue
        #     self.image_table = self.merge_mds(all_mds)

        if os.path.exists(os.path.join(pth,md_name)):
            self.image_table = self.load_metadata(join(pth), fname=md_name,try_shortcut=try_shortcut)
        else:
            all_mds = []
            for directory in os.listdir(pth):
                if os.path.exists(os.path.join(pth,directory,md_name)):
                    all_mds.append(self.load_metadata(os.path.join(pth,directory), fname=md_name,try_shortcut=try_shortcut))
            self.image_table = self.merge_mds(all_mds)
        # Future compability for different load types (e.g. remote vs local)
        if load_type=='local':
            self._open_file=self._read_local
        elif load_type=='google_cloud':
            raise NotImplementedError("google_cloud loading is not implemented.")
        # # Handle columns that don't import from text well
        # try:
        #     self.convert_data('XY', float)
        #     # if 'XYbefore' in list(self.image_table.columns):
        #     #     self.convert_data('XYbefore', float)
        #     # if 'XYbeforeTransform' in list(self.image_table.columns):
        #     #     self.convert_data('XYbeforeTransform', float)
        #     # if 'linescan' in list(self.image_table.columns):
        #     #     self.convert_data('linescan', float)
        # except Exception as e:
        #     self.image_table['XY'] = [literal_eval(i) for i in self.image_table['XY']]
        #     self.image_table['XYbeforeTransform'] = [literal_eval(i) for i in self.image_table['XYbeforeTransform']]
            
            
    @property
    def posnames(self):
        return self.image_table.Position.unique()
    @property
    def hybenames(self):
        acqs = self.image_table.acq.unique()
        hybes = [i.split('_')[0] for i in acqs]
        hybenames = []
        for name in hybes:
            if 'hybe' in name:
                hybenames.append(name)
        return hybenames
    @property
    def md(self):
        return self.image_table
    @property
    def channels(self):
        return self.image_table.Channel.unique()
    @property
    def Zindexes(self):
        return self.image_table.Zindex.unique()
    @property
    def acqnames(self):
        return self.image_table.acq.unique()
    
    def convert_data(self, column, dtype, isnan=np.nan):
        converted = []
        arr = self.image_table[column].values
        for i in arr:
            if isinstance(i, str):
                i = np.array(list(map(dtype, i.split())))
                converted.append(i)
            else:
                converted.append(i)
        self.image_table[column] = converted

    def load_metadata(self, pth, fname='Metadata.txt', delimiter='\t',try_shortcut=True):
        """
        Helper function to load a text metadata file.
        """
        shortcut = False
        if try_shortcut:
            # Look for a pkl file to load faster
            pkl_pth = join(pth, fname.split('.')[0]+'.pkl')
            if os.path.exists(pkl_pth):
                try:
                    md = pickle.load(open(pkl_pth,'rb'))
                    shortcut = True
                except:
                    print('Shortcut Failed')
                    print(pkl_pth)
                    shortcut = False
        if not shortcut:
            def convert(val):
                return np.array(list(map(float, val.split())))
            if self.low_memory:
                usecols = ['Channel', 'Exposure', 'Position', 'Scope', 'XY', 'Z', 'Zindex', 'acq', 'filename','TimestampFrame']
                md = pandas.read_csv(join(pth, fname), delimiter=delimiter,usecols=usecols,converters={'XY':convert})
            else:
                md = pandas.read_csv(join(pth, fname), delimiter=delimiter,converters={'XY':convert})
            md['root_pth'] = md.filename
            if pth[-1]!='/':
                pth = pth+'/'
            md.filename = pth + md.filename#[join(pth, f) for f in md.filename]
            if try_shortcut:
                # dump pickle for faster loading next time
                try:
                    pickle.dump(md,open(pkl_pth,'wb'))
                except Exception as e:
                    print(pkl_pth,e)
        return md
    
#     def update_metadata(self,acqs='All',fname='Metadata.txt', delimiter='\t'):
#         """
#         Helper function to update a text metadata file.
#         """
#         if acqs == 'All':
#             for acq in self.image_table.acq.unique():
#                 self.image_table[self.image_table.acq==acq].to_csv(join(pth, fname),sep=delimiter,index='False')
#         else:
#             for acq in acqs:
#                 self.image_table[self.image_table.acq==acq].to_csv(join(pth, fname),sep=delimiter,index='False')
        
    def merge_mds(self, mds):
        """
        Merge to metadata tables.
        
        WARNING: Not sophisticated enough to check for any duplicated information.
        """
        if not isinstance(mds, list):
            raise ValueError("mds argument must be a list of pandas image tables")
        merge_mds = pandas.concat(mds, ignore_index=True)
        return merge_mds
        # og_md = mds[0]
        # for md in mds[1:]:
        #     og_md = og_md.append(md, ignore_index=True,sort=True)
        # return og_md
        
#     def codestack_read(self, pos, z, bitmap, hybe_names=['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6', 'hybe7', 'hybe8', 'hybe9'], fnames_only=False):
#         """
#         Wrapper to load seqFISH images.
#         """
#         hybe_ref = 1
#         seq_name, hybe, channel = bitmap[0]
#         stk = [self.stkread(Position=pos, Zindex=z, hybe=hybe, 
#                                Channel=channel, fnames_only=True)[pos][0]]
#         for seq_name, hybe, channel in bitmap[1:]:
#             stk.append(self.stkread(Position=pos, Zindex=z, hybe=hybe, 
#                                    Channel=channel, fnames_only=True)[pos][0])
#         if fnames_only:
#             return [i for i in stk]
#         else:
#             return self._open_file({pos: [i for i in stk]})
            
    def stkread(self, groupby='Position', sortby=None,
                fnames_only=False, metadata=False, ffield=False, verbose=False,**kwargs):
        """
        Main interface of Metadata
        
        Parameters
        ----------
        groupby : str - all images with the same groupby field with be stacked
        sortby : str, list(str) - images in stks will be ordered by this(these) fields
        fnames_only : Bool (default False) - lazy loading
        metadata : Bool (default False) - whether to return metadata of images
        
        kwargs : Property Value pairs to subset images (see below)
        
        Returns
        -------
        stk of images if only one value of the groupby_value
        dictionary (groupby_value : stk) if more than one groupby_value
        stk/dict, metadata table if metadata=True
        fnames if fnames_only true
        fnames, metadata table if fnames_only and metadata
        
        Implemented kwargs
        ------------------
        Position : str, list(str)
        Channel : str, list(str)
        Zindex : int, list(int)
        acq : str, list(str)
        hybe : str, list(str)
        exposure : int, list(int)
        """
        # Input coercing
        self.verbose = verbose
        for key, value in kwargs.items():
            if not isinstance(value, list):
                kwargs[key] = [value]
        mask = np.full((self.image_table.shape[0],),True,dtype=bool)
        

        # Filter images according to some criteria
        if 'Position' in kwargs:
            mask = np.logical_and(mask,self.image_table['Position'].isin(kwargs['Position']))
            # image_subset_table = image_subset_table[image_subset_table['Position'].isin(kwargs['Position'])]
        if 'Channel' in kwargs:
            mask = np.logical_and(mask,self.image_table['Channel'].isin(kwargs['Channel']))
            # image_subset_table = image_subset_table[image_subset_table['Channel'].isin(kwargs['Channel'])]
        if 'acq' in kwargs:
            mask = np.logical_and(mask,self.image_table['acq'].isin(kwargs['acq']))
            # image_subset_table = image_subset_table[image_subset_table['acq'].isin(kwargs['acq'])]
        if 'Zindex' in kwargs:
            zindexes = kwargs['Zindex']
            if 'range' == kwargs['Zindex'][0]:
                ran,zmin,zmax = kwargs['Zindex']
                zindexes = range(zmin,zmax)
            mask = np.logical_and(mask,self.image_table['Zindex'].isin(zindexes))
            # image_subset_table = image_subset_table[image_subset_table['Zindex'].isin(zindexes)]
        if 'TimestampFrame' in kwargs:
            mask = np.logical_and(mask,self.image_table['TimestampFrame'].isin(kwargs['TimestampFrame']))
            # image_subset_table = image_subset_table[image_subset_table['TimestampFrame'].isin(kwargs['TimestampFrame'])]
        if 'hybe' in kwargs:
            hybes = self.image_table['acq'].str.split('_').str[0]
            mask = mask = np.logical_and(mask,hybes.isin(kwargs['hybe']))
            # keepers = []
            # for i in hybes:
            #     if i in kwargs['hybe']:
            #         keepers.append(True)
            #     else:
            #         keepers.append(False)
            # image_subset_table = image_subset_table[keepers]
        
        image_subset_table = self.image_table.loc[mask]

        # Group images and sort them then extract filenames of sorted images
        if sortby is not None: 
            image_subset_table.sort_values(sortby, inplace=True)
        image_groups = image_subset_table.groupby(groupby)
        fnames_output = {}
        mdata = {}
        for posname in image_groups.groups.keys():
            fnames_output[posname] = image_subset_table.loc[image_groups.groups[posname]].filename.values
            mdata[posname] = image_subset_table.loc[image_groups.groups[posname]]
        # Clunky block of code below allows getting filenames only, and handles returning 
        # dictionary if multiple groups present or ndarray only if single group
        if fnames_only:
            if len(list(fnames_output.keys()))==1:
                fnames_output = fnames_output[posname]
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                return fnames_output, mdata
            else:
                return fnames_output
        else:
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                    return self._open_file(fnames_output)[posname], mdata
                else:
                    return self._open_file(fnames_output), mdata
            else:
                stk = self._open_file(fnames_output) 
                if len(list(stk.keys()))==1:
                    return stk[posname]
                else:
                    return stk
    #def save_images(self, images, fname = '/Users/robertf/Downloads/tmp_stk.tif'):
    #    with TiffWriter(fname, bigtiff=False, imagej=True) as t:
    #        if len(images.shape)>2:
    #            for i in range(images.shape[2]):
    #                t.save(img_as_uint(images[:,:,i]))
    #        else:
    #            t.save(img_as_uint(images))
    #    return fname
        
    def _read_local(self, filename_dict, ffield=False, verbose=False):
        """
        Load images into dictionary of stks.
        """
        images_dict = {}
        for key, value in filename_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk
            arr = cv2.imread(join(value[0]),-1); 
            imgs = numpy.ndarray((numpy.size(value), numpy.size(arr,0), 
                                  numpy.size(arr,1)),arr.dtype)
            for img_idx, fname in enumerate(value):
                # Weird print style to print on same line
                if self.verbose:
                    sys.stdout.write("\r"+'opening '+path.split(fname)[-1])
                    sys.stdout.flush()
                img = cv2.imread(join(fname),-1)
                if ffield:
                    img = self.doFlatFieldCorrection(img, fname)
                imgs[img_idx,:,:]=img
                img_idx+=1
            # Best performance has most frequently indexed dimension first we're 
            # reordering here to maintain convention of Wollman lab
            images_dict[key] = imgs.transpose([1,2,0])          
            if verbose:
                print('Loaded {0} group of images.'.format(key))
            #from IPython.core.debugger import Tracer; Tracer()()
        return images_dict



        
#     def doFlatfieldCorrection(self, img, flt, **kwargs):
#         """
#         Perform flatfield correction.
        
#         Parameters
#         ----------
#         img : numpy.ndarray
#             2D image of type integer
#         flt : numpy.ndarray
#             2D image of type integer with the flatfield
#         """
#         print("Not implemented well. Woulnd't advise using")
#         cameraoffset = 100./2**16
#         bitdepth = 2.**16
#         flt = flt.astype(np.float32) - cameraoffset
#         flt = np.divide(flt, np.nanmean(flt.flatten()))
        
#         img = np.divide((img-cameraoffset).astype(np.float32), flt+cameraoffset)
#         flat_img = img.flatten()
#         rand_subset = np.random.randint(0, high=len(flat_img), size=10000)
#         flat_img = flat_img[rand_subset]
#         flat_img = np.percentile(flat_img, 1)
#         np.place(img, flt<0.05, flat_img)
#         np.place(img, img<0, 0)
#         np.place(img, img>bitdepth, bitdepth)
#         return img

# from numba import jit
# @jit(nopython = True)
# def DownScale(imgin): #use 2x downscaling for scrol speed   
#         #imgout = trans.downscale_local_mean(imgin,(Sc, Sc))
#     imgout = (imgin[0::2,0::2]+imgin[1::2,0::2]+imgin[0::2,1::2]+imgin[1::2,1::2])/4
#     return imgout

# def stkshow(data):
#     from pyqtgraph.Qt import QtCore, QtGui
#     import pyqtgraph as pg
#     import sys
#     import skimage.transform as trans

    
#     # determine if you need to start a Qt app. 
#     # If running from Spyder, the answer is a no.
#     # From cmd, yes. From nb, not sure actually.
#     if not QtGui.QApplication.instance():
#         app = QtGui.QApplication([])
#     else:
#         app = QtGui.QApplication.instance()
        
#     ## Create window with ImageView widget
#     win = QtGui.QMainWindow()
#     win.resize(680,680)
#     imv = pg.ImageView()
#     win.setCentralWidget(imv)
#     win.show()
#     win.setWindowTitle('Fetching image stack...')
    
    
    
    
#     resizeflg = 0;
#     maxxysize = 800;
#     maxdataxySc = np.floor(max(data.shape[0],data.shape[1])/maxxysize).astype('int')
#     if maxdataxySc>1:
#         resizeflg = 1;

#     if len(data.shape)==4:#RGB assume xytc
#         if data.shape[3]==3 or data.shape[3]==4:
#             if resizeflg:
#                 dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc,1,1]))).astype('int'),dtype = 'uint16')    
#                 for i in range(0,data.shape[2]):
#                     for j in range(0,data.shape[3]):
#                         dataRs[:,:,i,j] = DownScale(data[:,:,i,j])
#                 dataRs = dataRs.transpose((2,0,1,3))
#             else:
#                 dataRs = data;
#                 dataRs = dataRs.transpose((2,0,1,3))
#         else:
#             sys.exit('color channel needs to be RGB or RGBA')
#     elif len(data.shape)==3:
#         if resizeflg:
#             dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc,1]))).astype('int'),dtype = 'uint16')    
#             for i in range(0,data.shape[2]):
#                 dataRs[:,:,i] = DownScale(data[:,:,i])
#             dataRs = dataRs.transpose([2,0,1])
#         else:
#             dataRs = data;
#             dataRs = dataRs.transpose([2,0,1])
                
#     elif len(data.shape)==2:
#         if resizeflg:
#             dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc]))).astype('int'),dtype = 'uint16')
#             dataRs = DownScale(data)
#         else:
#             dataRs = data;
#     else:
#         print('Data must be 2D image or 3D image stack')
    

    
#     # Interpret image data as row-major instead of col-major
#     pg.setConfigOptions(imageAxisOrder='row-major')
    

#     win.setWindowTitle('Stack')
    
#     ## Display the data and assign each frame a 
#     imv.setImage(dataRs)#, xvals=np.linspace(1., dataRs.shape[0], dataRs.shape[0]))

#     ##must return the window to keep it open
#     return win
#     ## Start Qt event loop unless running in interactive mode.
#     if __name__ == '__main__':
#         import sys
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
