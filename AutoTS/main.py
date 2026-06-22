from detector import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
detector = Detector(model_type="IS")

detector.onImage('1.jpg')
print('yes')