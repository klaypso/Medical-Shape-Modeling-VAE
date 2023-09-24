import numpy as np
from torch.nn import MSELoss
from scipy import ndimage
from skimage.transform import resize
import SimpleITK as sitk
import torch
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import os
import random
from torch.utils.data import Dataset
from copy import copy
import logging
from skimage import measure
from scipy.special import softmax
import re
import imageio
import json
from batchgenerators.transforms.spatial_transforms import SpatialTransform, augment_spatial

def Tag(temp,index,tempC,num_component):
    tempC[index[0],index[1],index[2]]=num_component
    expand = []
    expand.append(index)
    count=1
    while len(expand)>0:
        temp_index = expand.pop()
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    new_index = [min(max(temp_index[0]+i,0),temp.shape[0]-1),min(max(temp_index[1]+j,0),temp.shape[1]-1),min(max(temp_index[2]+k,0),temp.shape[2]-1)]
                    if temp[new_index[0],new_index[1],new_index[2]]==1 and tempC[new_index[0],new_index[1],new_index[2]]==0:
                        expand.append(new_index)
                        tempC[new_index[0],new_index[1],new_index[2]]=num_component
                        count+=1
                        #print(count)

    return 
def check_connection(tumor_index,image):

    L = tumor_index.shape[0]
    temp = np.zeros_like(image)
    tempC = np.zeros_like(image)
    for i in range(L):
        temp[tumor_index[i,0],tumor_index[i,1],tumor_index[i,2]] = 1
    
    num_component = 0
    for i in range(L):

        if tempC[tumor_index[i,0],tumor_index[i,1],tumor_index[i,2]]==0:
            num_component += 1
            Tag(temp,tumor_index[i,:],tempC,num_component)
    cc=[]
    for i in range(L): 
        cc.append(tempC[tumor_index[i,0],tumor_index[i,1],tumor_index[i,2]])
    cc = np.array(cc)
    
    return cc



class BaseDataset(Dataset):
    """
    Base dataset class. Expects a list of dictionaries and a set of transforms
    to load data and transform it
    """
    def __init__(self, listdict, transforms=None):

        assert(type(listdict) == list), "Must provide a list of dicts to listdict"

        self._listdict = listdict
        self._transforms = transforms
        logging.debug('Dataset initialized with transform {}'.format(self._transforms))


    def __len__(self):
        return len(self._listdict)



    def __getitem__(self, idx):

        # here we assume the list dict is paths or image labels, we copy so as not
        # to modify the original list
        sample = copy(self._listdict[idx])
        if self._transforms:
            sample = self._transforms(sample)

        return sample

class BaseTransform(object):
    def __init__(self, fields):
        assert(isinstance(fields, (str, list))), "Fields must be a string or a list of strings"

        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields

    def __call__(self, sample):
        assert(isinstance(sample, dict)), "Each sample must be a dict"


class CopyField(BaseTransform):
    """
    Copy one field to another
    """

    def __init__(self, fields, to_field):
        super().__init__(fields)
        if len(fields) != 1:
            raise ValueError("Only provide one field for source")

        if isinstance(to_field, list):
            if len(to_field) != 1:
                raise ValueError("Only provide one field for destination")
        else:
            to_field = [to_field]

        self.to_field = to_field

    def __call__(self, data_dict):
        data_dict[self.to_field[0]] = copy(data_dict[self.fields[0]])

        return data_dict


class NiiLoader(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """
    def __init__(self, fields, root_dir='/', dtype=np.float32,pre_set=None):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.pre_set = pre_set
    def __call__(self, data_dict):
        if not self.pre_set:
            self.pre_set = random.sample(self.fields,2)

        data_dict['source'] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[0]]))).astype(self.dtype)
        data_dict['target'] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[1]]))).astype(self.dtype)
        data_dict['source_lung'] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[0]+'_lung']))).astype(self.dtype)
        data_dict['target_lung'] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[1]+'_lung']))).astype(self.dtype)
        if data_dict.get(self.pre_set[0]+'_pancreas',None) and data_dict.get(self.pre_set[1]+'_pancreas',None):
            data_dict['source_pancreas'] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[0]+'_pancreas']))).astype(self.dtype)
            data_dict['target_pancreas'] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, data_dict[self.pre_set[1]+'_pancreas']))).astype(self.dtype)
        return data_dict
class ReadNPY(BaseTransform):
    def __init__(self, root_dir='/',root_dir2='/',root_dir3=None):
        super().__init__(fields='sss')
        self.root_dir = root_dir
        self.root_dir2 = root_dir2
        self.root_dir3 = root_dir3
    def __call__(self, name):
        out_dict={}
        A=np.load(os.path.join(self.root_dir,name))
        if not os.path.exists(os.path.join(self.root_dir2,name[0:-4]+'.npy')):
            B=np.zeros_like(A['labelV'])
            B[...]=A['labelV'][...]
            B[B>0]=13
        else:
            B =np.load(os.path.join(self.root_dir2,name[0:-4]+'.npy'))
        if self.root_dir3 is not None:
            qihang_A=np.load(os.path.join(self.root_dir3,name))
            
        out_dict['labelV']=A['labelV'].astype(np.float32)
        out_dict['multi_organV']=B.astype(np.float32)
        out_dict['name']=name
        if self.root_dir3 is  not None:
            out_dict['softV']=qihang_A['pred'].astype(np.float32)
        else:
            out_dict['predV']=A['predV'].astype(np.float32)
            out_dict['softV']=np.argmax(out_dict['predV'],0).astype(np.float32)
        print('read done')
        return out_dict

class NumpyLoader(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', dtype=np.float32,pre_set=None,load_mask=False):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.pre_set = pre_set
        self.load_mask = load_mask
    def __call__(self, data_dict):
        if self.pre_set is None:
            self.pre_set = random.sample(self.fields,min(2,len(self.fields)))
        if data_dict.get(self.pre_set[0],None):
            data_dict['source'] = np.load(os.path.join(self.root_dir, data_dict[self.pre_set[0]])).astype(self.dtype)
        if self.load_mask:
            if data_dict.get(self.pre_set[0]+'_lung',None):
                data_dict['source_lung'] = np.load(os.path.join(self.root_dir, data_dict[self.pre_set[0]+'_lung'])).astype(self.dtype)
            if data_dict.get(self.pre_set[0]+'_pancreas',None): 
                data_dict['source_pancreas'] = (np.load(os.path.join(self.root_dir, data_dict[self.pre_set[0]+'_pancreas']))).astype(self.dtype)
        if len(self.pre_set)>1:
            if data_dict.get(self.pre_set[1],None):
                data_dict['target'] = np.load(os.path.join(self.root_dir, data_dict[self.pre_set[1]])).astype(self.dtype)
            if self.load_mask:
                if data_dict.get(self.pre_set[1]+'_lung',None):
                    data_dict['target_lung'] = np.load(os.path.join(self.root_dir, data_dict[self.pre_set[1]+'_lung'])).astype(self.dtype)
                if data_dict.get(self.pre_set[1]+'_pancreas',None):
                    data_dict['target_pancreas'] = (np.load(os.path.join(self.root_dir, data_dict[self.pre_set[1]+'_pancreas']))).astype(self.dtype)
        
        return data_dict
        
class CropResize(BaseTransform):
    def __init__(self, fields, output_size,pad=32, shift=0):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.output_size = output_size
        self.pad = pad
        self.shift = shift
    def __call__(self,data_dict):
        #pad_width=32
        shift = self.shift
        for f in self.fields:
            if data_dict.get(f) is not None:
                if isinstance(data_dict.get(f+'_pancreas_pred') ,np.ndarray):
                    pred = data_dict.get(f+'_pancreas_pred')
                    index = np.array(np.where(pred>0)).T
                    bbox_max = np.max(index,0)
                    bbox_min = np.min(index,0)
                    center = (bbox_max+bbox_min)//2
                    L = np.max(bbox_max-bbox_min)
                    pad_width = int(L*0.1)
                    pred = pred[max(center[0]-L//2-pad_width,0):min(center[0]+L//2+pad_width,pred.shape[0]), \
                            max(center[1]-L//2-pad_width,0):min(center[1]+L//2+pad_width,pred.shape[1]), \
                            max(center[2]-L//2-pad_width,0):min(center[2]+L//2+pad_width,pred.shape[2])]
                    diff = list(L+pad_width*2-np.array(pred.shape))
                    axis_pad_width = [(int(cur_diff/2), cur_diff-int(cur_diff/2)) for cur_diff in diff]
                    pred = np.pad(pred,axis_pad_width)
                    
                    data_dict[f+'_pancreas_pred']=resize(pred,self.output_size,order=0,anti_aliasing=False)
                else:
                    index = np.array(np.where(data_dict[f+'_pancreas']>0)).T
                    # print("pancreas: ", data_dict[f+'_pancreas'].shape)
                    # print("picture: ", data_dict[f].shape)
                    # print("index: ", index.shape)
                    if index.shape[0]>0:
                        bbox_max = np.max(index,0)
                        bbox_min = np.min(index,0)
                        center = (bbox_max+bbox_min)//2
                        L = np.max(bbox_max-bbox_min)
                        pad_width = int(L*0.1)
                    else:
                        center=np.array([64,64,64])
                        L=32
                        pad_width = int(L*0.1)
                img = data_dict.get(f)
                label = data_dict.get(f+'_pancreas')
                data_dict['ori_shape']=list(label.shape)
                label = label[max(center[0]-L//2-pad_width+shift,0):min(center[0]+L//2+pad_width+shift,label.shape[0]), \
                            max(center[1]-L//2-pad_width+shift,0):min(center[1]+L//2+pad_width+shift,label.shape[1]), \
                            max(center[2]-L//2-pad_width+shift,0):min(center[2]+L//2+pad_width+shift,label.shape[2])]
                diff = list(L+pad_width*2-np.array(label.shape))
                axis_pad_width = [(int(cur_diff/2), cur_diff-int(cur_diff/2)) for cur_diff in diff]
                
                label = np.pad(label,axis_pad_width)
                data_dict['ori_shape'] += list(label.shape)
                data_dict['ori_shape'] = np.array(data_dict['ori_shape'])
                img = img[max(center[0]-L//2-pad_width+shift,0):min(center[0]+L//2+pad_width+shift,img.shape[0]), \
                            max(center[1]-L//2-pad_width+shift,0):min(center[1]+L//2+pad_width+shift,img.shape[1]), \
                            max(center[2]-L//2-pad_width+shift,0):min(center[2]+L//2+pad_width+shift,img.shape[2])]
                diff = list(L+pad_width*2-np.array(img.shape))
                axis_pad_width = [(int(cur_diff/2), cur_diff-int(cur_diff/2)) for cur_diff in diff]
                img = np.pad(img,axis_pad_width)
                # print("1: ", data_dict[f].max(), data_dict[f].min())
                # print("2: ", data_dict[f].shape, self.output_size)
                data_dict[f]=resize(img,self.output_size)
                # print(self.output_size)
                # print("3: ", data_dict[f].max(), data_dict[f].min())
                data_dict[f+'_pancreas']=resize(label,self.output_size,order=0,anti_aliasing=False)

        return data_dict


class NumpyLoader_Multi(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', dtype=np.float32,load_mask=False,load_pred=False):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.dtype = dtype
        self.load_mask = load_mask
        self.load_pred = load_pred
    def __call__(self, data_dict):
        for f in self.fields:
            if data_dict.get(f) is not None:
                data_dict[f] = np.load(os.path.join(self.root_dir, data_dict[f])).astype(self.dtype)
            if self.load_mask:
                if data_dict.get(f+'_pancreas',None): 
                    data_dict[f+'_pancreas'] = np.load(os.path.join(self.root_dir, data_dict[f+'_pancreas'])).astype(self.dtype)
            if self.load_pred:
                if data_dict.get(f+'_pancreas_pred',None): 
                    data_dict[f+'_pancreas_pred'] = np.load(os.path.join(self.root_dir, data_dict[f+'_pancreas_pred'])).astype(self.dtype)
        return data_dict


class NumpyLoader_Multi_merge(BaseTransform):
    """
    Loads an image directly to np.array using npy files
    """

    def __init__(self, fields, root_dir='/', middle_path='/', dtype=np.float32,load_mask=False,load_pred=False,load_pseudo=False,load_seg_npy=False,mask_index=None):
        """
        Args:
            fields: fields specifying image paths to load
            root_dir: root dir of images
            dtype: resulting dtype of the loaded np.array, default is np.float32
        """
        super().__init__(fields)
        self.root_dir = root_dir
        self.middle_path = middle_path
        self.dtype = dtype
        self.load_mask = load_mask
        self.load_pred = load_pred
        self.load_pseudo = load_pseudo
        self.load_seg_npy = load_seg_npy
        self.mask_index = mask_index
    def __call__(self, input_string):
        data_dict={}
        data_dict['id'] = ''.join(re.findall(r'\d+', input_string))
        
        # score = json.load(open("/mnt/sdd/yaoyuan/VAE_segmentation/MaskData/score.json", 'r'))
        # aa = data_dict['id']
        # filename = os.path.join('compare/', f'{aa}_recon.pt')
        # torch.save(batch[label_key+'_recon_pred'][0], filename)
        
        # print(data_dict['id'])
        for f in self.fields:
            merge_data = np.load(os.path.join(self.root_dir, input_string))
            # if self.load_seg_npy:
            #     data_dict[f] = merge_data[0,1,...].astype(self.dtype)
            #     data_dict[f+'_pancreas'] = merge_data[0,1,...].astype(self.dtype)
            #     data_dict[f+'_score'] = np.array([score[data_dict['id']]])
            #     continue
            data_dict[f] = merge_data[...,0].astype(self.dtype)
            if self.load_mask:
                if self.mask_index is None:
                    data_dict[f+'_pancreas'] = merge_data[...,1].astype(self.dtype)
                else:
                    data_dict[f+'_pancreas'] = np.zeros_like(merge_data[...,1])
                    for label in self.mask_index:
                        if not isinstance(label[0], list): label[0] = [label[0]]
                        for lab in label[0]:
                            data_dict[f+'_pancreas'][merge_data[...,1]==lab]=label[1]
                    data_dict[f+'_pancreas'] = data_dict[f+'_pancreas'].astype(self.dtype)
                    # print(data_dict[f+'_pancreas'].shape)
            if self.load_pseudo:
                filename = os.path.join(self.middle_path, '{}_pred.npy'.format(data_dict['id']))
                data_dict[f+'_pancreas_pseudo'] = np.load(filename)
                # may introduce bug here
                # data_dict[f+'_pancreas_pseudo'] = data_dict[f+'_pancreas_pseudo'].cpu().detat