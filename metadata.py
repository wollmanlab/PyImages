# Metadata imports
from os import walk, listdir, path
from os.path import join, isdir
import sys

import pandas
import numpy
import numpy as np
from ast import literal_eval

from skimage import img_as_float, img_as_uint, io

#stkshow imports

class Metadata(object):
    def __init__(self, pth, md_name='Metadata.txt', load_type='local'):
        """
        Load metadata files.
        """
        self.base_pth = pth;
        # short circuit recursive search for metadatas if present in the top directory of 
        # the supplied pth.
        if md_name in listdir(pth):
            self.image_table = self.load_metadata(join(pth), fname=md_name)
        # Recursively append all metadatas in children directories of pth
        else:
            all_mds = []
            for subdir, curdir, filez in walk(pth):
                if md_name in filez:
                    try:
                        all_mds.append(self.load_metadata(join(pth, subdir), fname=md_name))
                    except:
                        continue
            self.image_table = self.merge_mds(all_mds)
        # Future compability for different load types (e.g. remote vs local)
        if load_type=='local':
            self._open_file=self._read_local
        elif load_type=='google_cloud':
            raise NotImplementedError("google_cloud loading is not implemented.")
        # Handle columns that don't import from text well
        try:
            self.convert_data('XY', float)
            if 'XYbefore' in list(self.image_table.columns):
                self.convert_data('XYbefore', float)
            if 'XYbeforeTransform' in list(self.image_table.columns):
                self.convert_data('XYbeforeTransform', float)
            if 'linescan' in list(self.image_table.columns):
                self.convert_data('linescan', float)
        except Exception as e:
            self.image_table['XY'] = [literal_eval(i) for i in self.image_table['XY']]
            self.image_table['XYbeforeTransform'] = [literal_eval(i) for i in self.image_table['XYbeforeTransform']]
            
            
    @property
    def posnames(self):
        return self.image_table.Position.unique()
    @property
    def hybenames(self):
        return self.image_table.hybe.unique()
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

    def load_metadata(self, pth, fname='Metadata.txt', delimiter='\t'):
        """
        Helper function to load a text metadata file.
        """
        md = pandas.read_csv(join(pth, fname), delimiter=delimiter)
        md['root_pth'] = md.filename
        md.filename = [join(pth, f) for f in md.filename]
        return md
        
    def merge_mds(self, mds):
        """
        Merge to metadata tables.
        
        WARNING: Not sophisticated enough to check for any duplicated information.
        """
        if not isinstance(mds, list):
            raise ValueError("mds argument must be a list of pandas image tables")
        og_md = mds[0]
        for md in mds[1:]:
            og_md = og_md.append(md, ignore_index=True)
        return og_md
        
    def codestack_read(self, pos, z, bitmap, hybe_names=['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6', 'hybe7', 'hybe8', 'hybe9'], fnames_only=False):
        """
        Wrapper to load seqFISH images.
        """
        hybe_ref = 1
        seq_name, hybe, channel = bitmap[0]
        stk = [self.stkread(Position=pos, Zindex=z, hybe=hybe, 
                               Channel=channel, fnames_only=True)[pos][0]]
        for seq_name, hybe, channel in bitmap[1:]:
            stk.append(self.stkread(Position=pos, Zindex=z, hybe=hybe, 
                                   Channel=channel, fnames_only=True)[pos][0])
        if fnames_only:
            return [i for i in stk]
        else:
            return self._open_file({pos: [i for i in stk]})
            
    def stkread(self, groupby='Position', sortby='TimestampFrame',
                fnames_only=False, metadata=False, ffield=False, **kwargs):
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
        """
        # Input coercing
        for key, value in kwargs.items():
            if not isinstance(value, list):
                kwargs[key] = [value]
        image_subset_table = self.image_table
        # Filter images according to some criteria
        if 'Position' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['Position'].isin(kwargs['Position'])]
        if 'Channel' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['Channel'].isin(kwargs['Channel'])]
        if 'acq' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['acq'].isin(kwargs['acq'])]
        if 'Zindex' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['Zindex'].isin(kwargs['Zindex'])]
        if 'TimestampFrame' in kwargs:
            image_subset_table = image_subset_table[image_subset_table['TimestampFrame'].isin(kwargs['TimestampFrame'])]
        if 'hybe' in kwargs:
            acqs = image_subset_table['acq']
            hybes = [i.split('_')[0] for i in acqs]
            keepers = []
            for i in hybes:
            	if i in kwargs['hybe']:
            		keepers.append(True)
            	else:
            		keepers.append(False)
            image_subset_table = image_subset_table[keepers]
        # Group images and sort them then extract filenames of sorted images
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
    def save_images(self, images, fname = '/Users/robertf/Downloads/tmp_stk.tif'):
        with TiffWriter(fname, bigtiff=False, imagej=True) as t:
            if len(images.shape)>2:
                for i in range(images.shape[2]):
                    t.save(img_as_uint(images[:,:,i]))
            else:
                t.save(img_as_uint(images))
        return fname
        
    def _read_local(self, filename_dict, ffield=False, verbose=False):
        """
        Load images into dictionary of stks.
        """
        images_dict = {}
        for key, value in filename_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk
            arr = io.imread(join(value[0])); 
            imgs = numpy.ndarray((numpy.size(value), numpy.size(arr,0), 
                                  numpy.size(arr,1)),arr.dtype)
            for img_idx, fname in enumerate(value):
                # Weird print style to print on same line
                sys.stdout.write("\r"+'opening '+path.split(fname)[-1])
                sys.stdout.flush()
                img = io.imread(join(fname))
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



        
    def doFlatfieldCorrection(self, img, flt, **kwargs):
        """
        Perform flatfield correction.
        
        Parameters
        ----------
        img : numpy.ndarray
            2D image of type integer
        flt : numpy.ndarray
            2D image of type integer with the flatfield
        """
        print("Not implemented well. Woulnd't advise using")
        cameraoffset = 100./2**16
        bitdepth = 2.**16
        flt = flt.astype(np.float32) - cameraoffset
        flt = np.divide(flt, np.nanmean(flt.flatten()))
        
        img = np.divide((img-cameraoffset).astype(np.float32), flt+cameraoffset)
        flat_img = img.flatten()
        rand_subset = np.random.randint(0, high=len(flat_img), size=10000)
        flat_img = flat_img[rand_subset]
        flat_img = np.percentile(flat_img, 1)
        np.place(img, flt<0.05, flat_img)
        np.place(img, img<0, 0)
        np.place(img, img>bitdepth, bitdepth)
        return img
