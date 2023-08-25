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
               