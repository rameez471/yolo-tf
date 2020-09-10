"""Class definaition of YOLO detection on image and video"""

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from PIL import Image,ImageDraw
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model

from model.yolo import yolo_eval,yolo_body
from model.utils import letter_box
