
from os import walk, listdir, path
from os.path import join, isdir
import sys

import pandas
import numpy

from skimage import img_as_float, img_as_uint, io

class Metadata(object):
    def __init__(self, pth, md_name='Metadata.txt', load_type='local'):
        """
        Load metadata files.
        """
        # short circuit recursive search for metadatas if present in the top directory of 
        # the supplied pth.
        if md_name in listdir(pth):
            self.image_table = self.load_metadata(join(pth), fname=md_name)
        # Recursively append all metadatas in children directories of pth
        else:
            all_mds = []
            for subdir, curdir, filez in walk(pth):
                if md_name in filez:
                    all_mds.append(self.load_metadata(join(pth, subdir), fname=md_name))
            self.image_table = self.merge_mds(all_mds)
        # Future compability for different load types (e.g. remote vs local)
        if load_type=='local':
            self._open_file=self._read_local
        elif load_type=='google_cloud':
            raise NotImplementedError("google_cloud loading is not implemented.")
        # Handle columns that don't import from text well
        self.image_table['XY'] = [map(float, i.split()) for i in self.image_table.XY.values]
        self.image_table['XYbefore'] = [map(float, i.split()) for i in self.image_table.XYbefore.values]
        
    def load_metadata(self, pth, fname='Metadata.txt', delimiter='\t'):
        md = pandas.read_csv(join(pth, fname), delimiter=delimiter)
        md.filename = [join(pth, f) for f in md.filename]
        return md
        
    def merge_mds(self, mds):
        if not isinstance(mds, list):
            raise ValueError("mds argument must be a list of pandas image tables")
        og_md = mds[0]
        for md in mds[1:]:
            og_md = og_md.append(md, ignore_index=True)
        return og_md
        
    def codestack_read(self, pos, z, bitmap, hybe_names=['hybe1', 'hybe2', 'hybe3', 'hybe4', 'hybe5', 'hybe6'], fnames_only=False):
        hybe_ref = 1
        seq_name, hybe, channel = bitmap[0]
        stk = [self.stkread(Position=pos, Zindex=z, hybe=hybe, 
                               Channel=channel, fnames_only=True)[pos][0]]
        for seq_name, hybe, channel in bitmap[1:]:
            stk.append(self.stkread(Position=pos, Zindex=z, hybe=hybe, 
                                   Channel=channel, fnames_only=True)[pos][0])
        if fnames_only:
            return [i[0] for i in stk]
        else:
            return self._open_file({pos: [i for i in stk]})
            
    def stkread(self, groupby='Position', sortby='TimestampFrame',
                fnames_only=False, metadata=False, **kwargs):
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
        if fnames_only:
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                return fnames_output, mdata
            else:
                return fnames_output
        else:
            if metadata:
                mdata = mdata[posname]
                return self._open_file(fnames_output), mdata
            else:
                return self._open_file(fnames_output) 
    # Would be good to not depend on tifffile since I've had problems installing it sometimes.
    def save_images(self, images, fname = '/Users/robertf/Downloads/tmp_stk.tif'):
        with TiffWriter(fname, bigtiff=False, imagej=True) as t:
            if len(images.shape)>2:
                for i in range(images.shape[2]):
                    t.save(img_as_uint(images[:,:,i]))
            else:
                t.save(img_as_uint(images))
        return fname
        
    def _read_local(self, filename_dict, verbose=False):
        images_dict = {}
        for key, value in filename_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk
            arr = io.imread(join(value[0])); 
            imgs = numpy.ndarray((numpy.size(value), numpy.size(arr,0), 
                                  numpy.size(arr,1)),arr.dtype)
            for img_idx, fname in enumerate(value):
                sys.stdout.write("\r"+'opening '+path.split(fname)[-1])
                sys.stdout.flush()
                #print('\r'+'opening '+fname); 
                imgs[img_idx,:,:]=io.imread(join(fname))
                img_idx+=1
            images_dict[key] = imgs.transpose([1,2,0])          
            if verbose:
                print('Loaded {0} group of images.'.format(key))
            #from IPython.core.debugger import Tracer; Tracer()()
        return images_dict