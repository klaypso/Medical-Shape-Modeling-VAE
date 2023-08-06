import numpy as np
import os.path as path
import nibabel as nib
from skimage.transform import resize
import os
import glob

image_path = '<path-to-the-image>/nih_data/Pancreas-CT/data' # TODO: modify this.
label_path = '<path-to-the-data>/nih_data/Pancreas-CT/TCIA_pancreas_labels-02-05-2017' # TODO: modify this.
to_path = 'data/nih' # TODO: modify this.
if not os.path.exists(to_path):
	os.makedirs(to_path)

names = glob.glob(path.join(image_path,'*.gz'))
names.sort()
names = [path.split(f)[1] for f in names]

pad = [32,32,32]
for img_name in names:
	label_name = 'label' + img_name.split('_')[1] # TODO: modify this.
	# label_name = 'label' + img_name.split('_')[0][5:8] # for synapse

	image = nib.load(path.join(image_path, img_name))
	spacing = image.affine[[0,1