import numpy as np
import sys, os
import argparse

# Make sure your "caffe/python" dir path is added to PYTHONPATH 
import caffe

def wb_extractor(deploy, cmodel):

    caffe.set_mode_gpu()

    #load caffe reference model
    caffenet = caffe.Net(str(deploy), str(cmodel), caffe.TEST)

    # generate dictionaries of the parameters
    params = caffenet.params.keys()
    source_params = {pr: (caffenet.params[pr][0].data, caffenet.params[pr][1].data) for pr in params}
    
    # Layer name, weights and bias from .caffemodel
    for pr in params:
        print ("Layer =" + str(pr))
        print ("weights" + str(source_params[pr][0]))
        print ("bias" + str(source_params[pr][1]))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path2deploy',
                        help="Path to deploy.prototxt")
    parser.add_argument('path2modelfile',
                        help="Path to .caffemodel")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    path2deploy = args.path2deploy
    path2modelfile = args.path2modelfile
    
    if not os.path.isfile(path2deploy):
       print ("Deploy prototxt not found ")
       sys.exit()

    if not os.path.isfile(path2modelfile):
       print ("Model file not found ")
       sys.exit()

    wb_extractor(path2deploy, path2modelfile)

