# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import math
import argparse
import numpy as np
import cntk
import _cntk_py
import cntk.io.transforms as xforms

from cntk.logging import *
from cntk.ops import *
from cntk.train.distributed import data_parallel_distributed_learner, Communicator
from cntk.io import ImageDeserializer, MinibatchSource, StreamDef, StreamDefs, FULL_DATA_SWEEP
from cntk.layers import Convolution2D, Activation, MaxPooling, Dense, Dropout, default_options, Sequential, For
from cntk.initializer import normal
from cntk.train.training_session import *
from cntk import input, placeholder, combine, alias, sequence, parameter, constant

# default Paths relative to current python file.
abs_path   = os.path.dirname(os.path.abspath(__file__))
#data_path  = os.path.join(abs_path, "..", "..", "..", "DataSets", "ImageNet")
DATA_LOCAL_PATH = "C:\\Users\\t-anik\\Desktop\\personal\\KaggleData\\cervical-cancer\\augm_data\\"
data_path  = DATA_LOCAL_PATH
model_path = os.path.join(abs_path, "Models")
log_dir = None

# model dimensions
image_height = 224
image_width  = 224
num_channels = 3  # RGB
num_classes  = 3
model_name   = "Cervix_VGG16.model"

# Create a minibatch source.
def create_image_mb_source(map_file, is_training, total_number_of_samples):
    if not os.path.exists(map_file):
        raise RuntimeError("File '%s' does not exist." %map_file)

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    '''if is_training:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.4375:0.875, jitter_type='uniratio') # train uses jitter
        ]
    else: 
        transforms += [
            xforms.crop(crop_type='center', side_ratio=0.5833333) # test has no jitter
        ]'''

    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
    ]

    # deserializer
    return MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
            labels   = StreamDef(field='label', shape=num_classes))),   # and second as 'label'
        randomize = is_training, 
        max_samples=total_number_of_samples,
        multithreaded_deserializer = True)

# Create the network.
def create_vgg16():

    # Input variables denoting the features and label data
    feature_var = cntk.input((num_channels, image_height, image_width))
    label_var = cntk.input((num_classes))

    # apply model to input
    # remove mean value 
    input = minus(feature_var, constant([[[104]], [[117]], [[124]]]), name='mean_removed_input')
    
    with default_options(activation=None, pad=True, bias=True):
        z = Sequential([
            # we separate Convolution and ReLU to name the output for feature extraction (usually before ReLU) 
            For(range(2), lambda i: [
                Convolution2D((3,3), 64, name='conv1_{}'.format(i)), 
                Activation(activation=relu, name='relu1_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool1'),

            For(range(2), lambda i: [
                Convolution2D((3,3), 128, name='conv2_{}'.format(i)), 
                Activation(activation=relu, name='relu2_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool2'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 256, name='conv3_{}'.format(i)), 
                Activation(activation=relu, name='relu3_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool3'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 512, name='conv4_{}'.format(i)), 
                Activation(activation=relu, name='relu4_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool4'),

            For(range(3), lambda i: [
                Convolution2D((3,3), 512, name='conv5_{}'.format(i)), 
                Activation(activation=relu, name='relu5_{}'.format(i)), 
            ]),
            MaxPooling((2,2), (2,2), name='pool5'),

            Dense(4096, name='fc6'), 
            Activation(activation=relu, name='relu6'), 
            Dropout(0.5, name='drop6'), 
            Dense(4096, name='fc7'), 
            Activation(activation=relu, name='relu7'), 
            Dropout(0.5, name='drop7'),
            Dense(num_classes, name='fc8')
            ])(input)

    # loss and metric
    ce = cntk.cross_entropy_with_softmax(z, label_var)
    pe = cntk.classification_error(z, label_var)
    pe5 = cntk.classification_error(z, label_var, topN=5)

    log_number_of_parameters(z) ; print()

    return {
        'feature': feature_var,
        'label': label_var,
        'ce' : ce,
        'pe' : pe,
        'pe5': pe5, 
        'output': z
    }

# Create trainer
def create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer):
    # Set learning parameters
    lr_per_mb         = [0.01]*20 + [0.001]*20 + [0.0001]*20 + [0.00001]*10 + [0.000001]
    lr_schedule       = cntk.learning_rate_schedule(lr_per_mb, unit=cntk.learners.UnitType.minibatch, epoch_size=epoch_size)
    mm_schedule       = cntk.learners.momentum_schedule(0.9)
    l2_reg_weight     = 0.0005 # CNTK L2 regularization is per sample, thus same as Caffe
    
    # Create learner
    if block_size != None and num_quantization_bits != 32:
        raise RuntimeError("Block momentum cannot be used with quantization, please remove quantized_bits option.")

    # Create learner
    local_learner = cntk.learners.momentum_sgd(network['output'].parameters, lr_schedule, mm_schedule, unit_gain=False, l2_regularization_weight=l2_reg_weight)
    # Since we reuse parameter settings (learning rate, momentum) from Caffe, we set unit_gain to False to ensure consistency 
    if block_size != None:
        parameter_learner = cntk.train.distributed.block_momentum_distributed_learner(local_learner, block_size=block_size)
    else:
        parameter_learner = cntk.train.distributed.data_parallel_distributed_learner(local_learner, num_quantization_bits=num_quantization_bits, distributed_after=warm_up)

    # Create trainer
    return cntk.Trainer(network['output'], (network['ce'], network['pe']), parameter_learner, progress_printer)

# Train and test
def train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore):

    # define mapping from intput streams to network inputs
    input_map = {
        network['feature']: train_source.streams.features,
        network['label']: train_source.streams.labels
    }

    mb_size_schedule = cntk.minibatch_size_schedule(minibatch_size)

    # Train all minibatches 
    training_session(
        trainer=trainer, mb_source = train_source,
        model_inputs_to_streams = input_map,
        mb_size = mb_size_schedule,
        progress_frequency=epoch_size,
        checkpoint_config = CheckpointConfig(filename = os.path.join(model_path, model_name), restore=restore),
        test_config = TestConfig(source=test_source, mb_size=mb_size_schedule)
    ).train()

# Train and evaluate the network.
def vgg16_train_and_eval(train_data, test_data, num_quantization_bits=32,
                         minibatch_size=128, block_size=3200,
                         epoch_size = 1281167, max_epochs=80, warm_up=0,
                         restore=True, log_to_file=None, num_mbs_per_log=None, 
                         gen_heartbeat=False, tensorboard_logdir=None):
    _cntk_py.set_computation_network_trace_level(0)

    distributed_sync_report_freq = None
    if block_size is not None:
        distributed_sync_report_freq = 1

    progress_printer = [ProgressPrinter(
        freq=num_mbs_per_log,
        tag='Training',
        log_to_file=log_to_file,
        rank=Communicator.rank(),
        gen_heartbeat=gen_heartbeat,
        test_freq=num_mbs_per_log,
        num_epochs=max_epochs,
        distributed_freq=distributed_sync_report_freq)]

    if tensorboard_logdir is not None:
        progress_printer.append(cntk.logging.TensorBoardProgressWriter(
        freq=num_mbs_per_log,
        log_dir=tensorboard_logdir,
        rank=cntk.train.distributed.Communicator.rank(),
        model=network['output']))

    network = create_vgg16()
    trainer = create_trainer(network, epoch_size, num_quantization_bits, block_size, warm_up, progress_printer)
    train_source = create_image_mb_source(train_data, True, total_number_of_samples=max_epochs * epoch_size)
    test_source = create_image_mb_source(test_data, False, total_number_of_samples=FULL_DATA_SWEEP)
    train_and_test(network, trainer, train_source, test_source, minibatch_size, epoch_size, restore)
 

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-datadir', '--datadir', help='Data directory where the ImageNet dataset is located', required=False, default=data_path)
    parser.add_argument('-outputdir', '--outputdir', help='Output directory for checkpoints and models', required=False, default=None)
    parser.add_argument('-logdir', '--logdir', help='Log file', required=False, default=None)
    parser.add_argument('-tensorboard_logdir', '--tensorboard_logdir', help='Directory where TensorBoard logs should be created', required=False, default=None)
    parser.add_argument('-n', '--num_epochs', help='Total number of epochs to train', type=int, required=False, default='80')
    parser.add_argument('-m', '--minibatch_size', help='Minibatch size', type=int, required=False, default='128')
    parser.add_argument('-e', '--epoch_size', help='Epoch size', type=int, required=False, default='100')
    parser.add_argument('-q', '--quantized_bits', help='Number of quantized bits used for gradient aggregation', type=int, required=False, default='32')
    parser.add_argument('-a', '--distributed_after', help='Number of samples to train with before running distributed', type=int, required=False, default='0')
    parser.add_argument('-b', '--block_samples', type=int, help="Number of samples per block for block momentum (BM) distributed learner (if 0 BM learner is not used)", required=False, default=None)
    parser.add_argument('-r', '--restart', help='Indicating whether to restart from scratch (instead of restart from checkpoint file by default)', action='store_true')
    parser.add_argument('-device', '--device', type=int, help="Force to run the script on a specified device", required=False, default=None)

    args = vars(parser.parse_args())

    if args['outputdir'] is not None:
        model_path = args['outputdir'] + "/models"
    if args['datadir'] is not None:
        data_path = args['datadir']
    if args['logdir'] is not None:
        log_dir = args['logdir']
    if args['device'] is not None:
        cntk.device.try_set_default_device(cntk.device.gpu(args['device']))

    #if data_path == DATA_LOCAL_PATH:
    #    cntk.device.try_set_default_device(cntk.device.cpu())

    if not os.path.isdir(data_path):
        raise RuntimeError("Directory %s does not exist" % data_path)

    train_data=os.path.join(data_path, 'train_map.txt')
    test_data=os.path.join(data_path, 'test_map.txt')

    try:
        vgg16_train_and_eval(train_data, test_data, 
                             minibatch_size=args['minibatch_size'], 
                             epoch_size=args['epoch_size'],
                             num_quantization_bits=args['quantized_bits'],
                             block_size=args['block_samples'],
                             warm_up=args['distributed_after'],
                             max_epochs=args['num_epochs'],
                             restore=not args['restart'],
                             log_to_file=args['logdir'],
                             num_mbs_per_log=200,
                             gen_heartbeat=True,
                             tensorboard_logdir=args['tensorboard_logdir'])
    finally:
        cntk.train.distributed.Communicator.finalize()  