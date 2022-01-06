import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import os
import shutil
import random

#Image paths
projectPath = r'D:\chest_xray\train\PNEUMONIA'
outputdir = r'D:\chest_xray_randomly_oversample\train\PNEUMONIA'



files = [file for file in os.listdir(projectPath) if os.path.isfile(os.path.join(projectPath, file))]

# Amount of random files you'd like to select
random_amount = 996
for x in range(random_amount):
    selection = random.randint(0, len(files)-1)
    file = files.pop(selection)
    shutil.copyfile(os.path.join(projectPath, file), os.path.join(outputdir, file))