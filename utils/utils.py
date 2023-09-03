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
    def __init__(self, fields, root_dir='/', d