import caffe
import numpy as np
import argparse

def weights2caffemodel(path2weights, path2prototxt, path2caffemodel)
    caffe.set_mode_gpu()
    #To Use CPU, uncomment bottome line and delete above line
    #caffe.set_mode_cpu()

    net = caffe.Net(str(path2prototxt), str(path2caffemodel))

    
   

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path2weights',
                        help = "Path to darknet weights file")
    parser.add_argument('path2caffemodel',
                        help = "Path to output caffemodel file")
    parser.add_argument('path2prototxt', 
                        help = "Path to prototxt file")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    path2weights = args.path2weights
    path2caffemodel = args.path2caffemodel
    path2prototxt = args.path2prototxt

    if not os.path.isfile(path2weights):
       print ("Darknet weights file not found")
       sys.exit()

    if not os.path.isfile(path2prototxt):
       print ("Prototxt file not found")
       sys.exit()

    weights2caffemodel(path2weights, path2prototxt, path2caffemodel)
