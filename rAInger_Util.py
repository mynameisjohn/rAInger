# builtins
import os
import random

# Used to load images in parallel
import concurrent.futures

# OpenCV and numpy
import cv2
import numpy as np

# Tensorflow and tflearn
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# argparse
import argparse

# Store the args as a global within this module
args = None

# This function reads in input args and assigns the global args variable
def get_arguments():
    parser = argparse.ArgumentParser()

    # Image resize dimension
    parser.add_argument('--img_size', type=int, default=50, help='All training and test images will be resized to an IMG_SIZExIMG_SIZE image')

    # Training and Test directories
    parser.add_argument('--train_dir', type = str, default='train', help='Location of training file directory')
    parser.add_argument('--train_data_file', type = str, help='The cached training file data')
    parser.add_argument('--num_validate', type = int, default=500, help='The number of training files used for validation')
    parser.add_argument('--test_dir', type = str, default='test', help='Location of test file directory')
    parser.add_argument('--test_data_file', type = str, help='The cached test file data')

    # Net parameters
    parser.add_argument('--num_filters', type=int, default=32, help='The number of convolution filters used')
    parser.add_argument('--filter_size', type=int, default=5, help='The size of the convolution filters used')
    parser.add_argument('--num_neurons', type=int, default=1024, help='The count of fully connected neurons')
    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learn Rate of training algorithm')
    parser.add_argument('--n_epoch', type=int, default=10, help='Learn Rate')
    parser.add_argument('--model_file', type=str, help='The cached model file')

    # Clean cached files
    parser.add_argument('-ci', '--clean_images', type=bool, default=False, help='Delete any cached data files')
    parser.add_argument('-cm', '--clean_model', type=bool, default=False, help='Delete any cached data files')

    # Decision threshold
    # For the live camera mode, if the prediction value exceeds this threshold then it's a match
    parser.add_argument('-thresh', type=float, default=1.)

    # Camera mode
    # if this is on, opencv is used to open the default camera. 
    # When the space key is pressed an image is captured and used
    # for a prediction using the trained model
    # If this is off we use what's in the test directory
    parser.add_argument('--cam', type=bool, default=True, help='Use camera as prediction input')
    parser.add_argument('--motion', type=bool, default=False, help='Require motion for camera prediction')
    parser.add_argument('--motion_window', type=float, default=40., help="Duration in seconds of our motion detection window")
    parser.add_argument('--min_detect', type=int, default=500, help="minimum motion detection area size")

    # get arguments and cache them as a global variable
    global args
    args = parser.parse_args()
    return args

# Function to remove cached data
def rmFile(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

# See if the image starts with 'cat' or 'dog' 
# and give it a bool array
def create_label(image_name):
    word_label = image_name.split('.')[-3]
    if word_label=='cat':
        return np.array([1,0])
    elif word_label=='dog':
        return np.array([0,1])
    raise RuntimeError('Invalid image name')

# Make sure image is right size, grey, convert to np array
def fmt_img(img_data):
    ret = cv2.resize(img_data, (args.img_size, args.img_size))
    if len(ret.shape) > 2 and ret.shape[2] == 3:
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    return np.array(ret).reshape(args.img_size, args.img_size, 1)

# parallel executor function, loads an image as gray
# and formats it according to what tensorflow wants
def load_img(imgPath):
    img_data = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img_data is None:
        return
    return fmt_img(img_data)

# Create training or test data
def create_data(strDir, strFile, bTrain):
    # If the file exists, return it
    strFullPath = os.path.join(strDir, strFile)
    if os.path.isfile(strFullPath):
        print('Using file on disk', strFullPath)
        return np.load(strFullPath)

    # Split each image load task across multiple threads, then join and store data
    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Iterate over every .jpg file in the directory
        imgFiles = [img for img in os.listdir(strDir) if os.path.splitext(img)[1] == '.jpg']
        imgPaths = [os.path.join(strDir, img) for img in imgFiles]
        for img, img_data in zip(imgFiles, executor.map(load_img,(imgPaths))):
            if bTrain:
                # Use above label if training data
                data.append([img_data, create_label(img)])
            else:
                # Enumerate if testing data
                img_num = img.split('.')[0]
                data.append([img_data, img_num])
    # Randomize the data and save it to a file
    random.shuffle(data)
    np.save(strFullPath, data)
    return data

# function to create network, either for training or testing
# this network was not really designed by me - instead I followed
# the cat - dog detector tutorial below and used their network
# https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b
def create_net(bTrain):
    # Create layer graph
    # Each block here creates a new layer in the network
    tf.reset_default_graph()

    # Input image is a IMG_SIZE x IMG_SIZE gray image
    convnet = input_data(shape = [None, args.img_size, args.img_size, 1], name = 'input')

    # L1
    # max_pool is kind of like a downsampling operation
    # we do this as we progress through the layers and get smaller images
    # 'relu' is a relative unity activation - negative values go to zero
    convnet = conv_2d(convnet, args.num_filters, args.filter_size, activation = 'relu')
    convnet = max_pool_2d(convnet, args.filter_size)

    # L2, twice the number of filters and max_pool
    convnet = conv_2d(convnet, 2 * args.num_filters, args.filter_size, activation = 'relu')
    convnet = max_pool_2d(convnet, args.filter_size)

    # L3, another doubling of filter count
    convnet = conv_2d(convnet, 4 * args.num_filters, args.filter_size, activation = 'relu')
    convnet = max_pool_2d(convnet, args.filter_size)

    # Layers 4 and 5 seem to 'wind back' the filter count
    convnet = conv_2d(convnet, 2 * args.num_filters, args.filter_size, activation = 'relu')
    convnet = max_pool_2d(convnet, args.filter_size)
    convnet = conv_2d(convnet, args.num_filters, args.filter_size, activation = 'relu')
    convnet = max_pool_2d(convnet, args.filter_size)

    # A fully connected layer, which means every "neuron" (or pixel) 
    # in this layer is "connected" to every neuron in the previous layer
    # I think that means we're done downsampling and ready to evaluate
    # what the pixels in this layer's image are telling us
    convnet = fully_connected(convnet, args.num_neurons, activation = 'relu')

    # The dropout operation takes the previous layer and scales
    # it by 1/KEEP_THRESH - anything below KEEP_THRESH will
    # shrink - I'm not sure if things actually go to zero...
    KEEP_THRESH = 0.8
    convnet = dropout(convnet, KEEP_THRESH)

    # Create another fully connected layer
    if bTrain:
        # We use a softmax for training because it more quickly approaces the truth
        convnet = fully_connected(convnet, 2, activation='softmax')
        # The regression step is a gradient descent - I think the LR is the
        # step size, and I assume it gets followed until something goes to zero?
        convnet = regression(convnet, optimizer='adam', learning_rate=args.learn_rate, loss='categorical_crossentropy', name='targets')
    else:
        # We could use a softmax here, but it's more valuable
        # to know if we aren't seeing anything useful
        convnet = fully_connected(convnet, 2, activation='relu')

    return convnet

# OpenCV motion detector class
class cvMotionDetector:
    lastFrame = None
    def __init__(self, min_detect, blurKernDim = 21, dropThresh = 25):
        self.blurKernDim = (blurKernDim, blurKernDim)
        self.dropThresh = dropThresh

    def detect(self, frame, bDrawRect = False):
        blur = cv2.GaussianBlur(frame, self.blurKernDim, 0)
        if self.lastFrame is None:
            self.lastFrame = blur
            return False
        # Detect diff betwen frame and find contours
        deltaFrame = cv2.absdiff(lastFrame, blur)
        lastFrame = blur
        ret, thresh = cv2.threshold(deltaFrame, 25, 255, cv2.THRESH_BINARY)
        dilate = cv2.dilate(thresh, None, iterations = 2)
        _, contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Walk contours
        for c in contours:
            # If the contour area exceeds our detection threshold, return true
            if cv2.contourArea(c) > self.min_detect:
                # draw a rectangle around the contour if desired
                if bDrawRect:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                return True

        return False