"""Read Darknet config and wights and create model."""

import argparse
import configparser
import io
import os
from collections import defaultdict

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    ZeroPadding2D,
    Add,
    UpSampling2D,
    MaxPooling2D,
    Concatenate,
    LeakyReLU,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model as plot

parser = argparse.ArgumentParser(description='Darknet to Keras Converter')
parse.add_argument('config_path', help='Pah to Darknet cfg file')
parser.add_argument('weights_path', help='Path to Darknet weights file')
parser.add_argument('output_apth', help='Path to output model file')
parser.add_argument(
    '-p',
    '--plot_model',
    help='Path to generateed Keras model and save as image',
    action='store_true'
)

parser.add_argument(
    '-w',
    '--weights_only',
    help='Save as Keras weight file insted of model file.',
    action='store_true'
)

def unique_config_section(config_file):
    '''Convert all config section to have unique names.'''
    section_counters = defaultdict(int)
    output_steam = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_steam.write(line)
    output_steam.seek(0)
    return output_steam