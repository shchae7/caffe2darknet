import caffe
import numpy as np

from collections import OrderedDict
from cfg import *
from prototxt import *

caffe.set_mode_gpu()
caffe.set_device(1)

def extract_weights_from_caffemodel(protofile, caffemodel):
    net = caffe.Net(protofile, caffemodel, caffe.TEST)
    print(net.params['conv1'][0].data)


def extract_weights_from_caffemodelv2(protofile, caffemodel):
    model = parse_caffemodel(caffemodel)
    layers = model.layer
    if len(layers) == 0:
        print('Using V1LayerParameter')
        layers = model.layers

    lmap = {}
    for l in layers:
       lmap[l.name] = l
    
    net_info = parse_prototxt(protofile)
    props = net_info['props']

    wdata = []
    blocks = []
    block = OrderedDict()
    block['type'] = 'net'
    
    block['batch'] = props['input_dim'][0]
    block['channels'] = props['input_dim'][1]
    block['height'] = props['input_dim'][2]
    block['width'] = props['input_dim'][3]
    
    blocks.append(block)

    layers = net_info['layers']
    
    for i in range(len(layers)):
       print(layers[i])
       print('\n')

def save_weights(data, weightfile):
    print("Converting caffemodel to weights...")
    weights = np.zeros((wsize+4,), dtype=np.int32)
    weights[0] = 0
    weights[1] = 1
    weights[2] = 0
    weights[3] = 0

    weights.tofile(weightfile)
    weights = np.fromfile(weightfile, dtype=np.float32)
    weights[4:] = data
    weights.tofile(weightfile)

if __name__ == '__main__':
    import sys
 
    extract_weights_from_caffemodelv2('concat_deploy.prototxt', 'simpnet_iter_500.caffemodel')
