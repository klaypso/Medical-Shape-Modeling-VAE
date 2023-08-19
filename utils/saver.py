import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

class Saver():
    def __init__(self, display_dir,display_freq):
        self.display_dir = display_dir
        self.display_freq = display_freq
        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        # create tensorboard writer
        self.writer = SummaryW