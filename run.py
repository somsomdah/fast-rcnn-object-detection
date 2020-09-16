import os
import matplotlib.pyplot as plt
from utils import *

module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
detector = hub.load(module_handle).signatures['default']

for file in os.listdir('./inputs'):
    filename,extention=file.split('.')
    result=run_detector(detector, load_and_resize_image('./inputs/'+file))
    plt.imsave('./outputs/'+filename+'_result.'+extention,result)
