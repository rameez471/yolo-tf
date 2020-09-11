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
parser.add_argument('config_path', help='Pah to Darknet cfg file')
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


def main(args):
    config_path =  os.path.expanduser(args.config_path)
    weight_path = os.path.expanduser(args.weight_path)
    output_path = os.path.expanduser(args.output_path)

    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(
                                                                config_path)
    
    assert weight_path.endswith('.weights'), '{} is not a .weights file'.format(
                                                                weight_path)

    assert output_path.endswith('.h5'), '{} is not a .h5 file'.format(
                                                                output_path)

    output_root = os.path.splitext(output_path)[0]

    # Load weights and config
    print('Loading weights')
    weights_file = open(weight_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3,), dtype='int32',
                                        buffer=weights_file.read(12))
    
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype='int64',buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32',buffer=weights_file.read(4))

    print('weights Header: ',major, minor, revision,seen)

    print('Parsing Darknet Configurations')
    unique_config_file = unique_config_section(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Model')
    input_layer = Input(shape=(None,None,3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4

    count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('Parsing section {}',format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = int(cfg_parser[section]['activation'])
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            prev_layer_shape = K.shape(prev_layer)

            weight_shape = (size,size,prev_layer_shape[-1],filters)
            darknet_w_shape = (filters,weight_shape[2],size,size)
            weight_size = np.product(weight_shape)

            print('conv2d','bn' if batch_normalize else ' ',activation,weight_shape)

            conv_bias = np.ndarray(shape=(filters,),dtype='float32',buffer=weights_file.read(filter * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(shape=(3,filters),dtype='float32', buffer=weights_file.read(filter * 12))
                count += 3*filters

                bn_weight_list = [
                    bn_weights[0],
                    conv_bias,
                    bn_weights[1],
                    bn_weights[2]
                ]
            
            conv_weights = np.ndarray(shape=darknet_w_shape,dtype='float32', 
                                       buffer=weights_file.read_file(weight_size * 4))
            count += weight_size

            ### channel first to channel last
            conv_weights = np.transpose(conv_weights,[2,3,1,0])
            conv_weights = [conv_weights] if batch_normalize else [conv_weights,conv_bias]

            #Activation
            act_fn = None
            if activation == 'leaky':
                pass
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(activation,section))

            if stride > 1:
                prev_layer = ZeroPadding2D(((1,0),(1,0)))(prev_layer)
            conv_layer = (Conv2D(filters,(size,size),strides=(stride,stride),
                                kernel_regularizer=l2(weight_decay),
                                use_bias=not batch_normalize,
                                weights=conv_weights,activation=act_fn,
                                padding=padding))(prev_layer)   

            if batch_normalize:
                conv_layer = (BatchNormalization(
                    weights = bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer) 
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)
            
        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layer: ',layers)
                concatenation_layer = Concatenate()(layers)
                all_layers.append(concatenation_layer)
                prev_layer = concatenation_layer
            else:
                skip_layer = layers[0]
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            strides = int(cfg_parser[section]['strides'])
            all_layers.append(MaxPooling2D(pool_size=(size,size),
                                            strides=(strides,strides),
                                            padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported'
            all_layers.append(Add()([all_layers[index],prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2,' Only stride 2 supported'
            all_layers.append(UpSampling2D(strides)(prev_layer))
            prev_layer = all_layers[-1]
        
        elif section.startswith('yolo'):
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError('Unsupported section header type: {}'.format(section))

    #Create and save model
    if len(out_index) == 0:
        out_index.append(len(all_layers)-1)
    model = Model(inputs=input_layer,outputs=[all_layers[i] for i in out_index])
    print(model.summary())

    if args.weights_only:
        model.save_weights('{}'.format(output_path))
        print('Saved to {}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        print('Saved to {}'.format(output_path))

    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    if remaining_weights > 0:
        print('Read {} of {} from Darknet weights'.format(count,count + remaining_weights))
    
    if args.plot_model:
        plot(model,to_file='{}.png'.format(output_root),show_shapes=True)
        print('Saved model to plot to {}.png'.format(output_root))

if __name__ == '__main__':
    main(parser.parse_args())
