# Tensorflow / tflearn catdog example
# Unceremoniously lifted from this example from Medium.com, of all places...
# https://medium.com/@curiousily/tensorflow-for-hackers-part-iii-convolutional-neural-networks-c077618e590b
# builtins
import time
import os         
import random

# OpenCV and numpy
import cv2
import numpy as np

# For final plot
import matplotlib.pyplot as plt

# Tensorflow and tflearn
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# Used to load images in parallel
import concurrent.futures

# program input
import argparse

# Argument format is as follows
# tf_catdog 
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
parser.add_argument('--model_file', type=str, help='The cached model file')

# Clean cached files
parser.add_argument('-ci', '--clean_images', type=bool, default=False, help='Delete any cached data files')
parser.add_argument('-cm', '--clean_model', type=bool, default=False, help='Delete any cached data files')

# Camera mode
# if this is on, opencv is used to open the default camera. 
# When the space key is pressed an image is captured and used
# for a prediction using the trained model
# If this is off we use what's in the test directory
parser.add_argument('--cam', type=bool, default=True, help='Use camera as prediction input')
parser.add_argument('--min_detect', type=int, default=500, help="minimum motion detection area size")

# get arguments
args = parser.parse_args()

# directories for training and test data
TRAIN_DIR = args.train_dir
TEST_DIR = args.test_dir

# Image size (images are resized to this)
IMG_SIZE = args.img_size

# Net parameters
NUM_FILTERS = args.num_filters
FILTER_SIZE = args.filter_size
NUM_NEURONS = args.num_neurons

# Training and test data file storage
if args.train_data_file is None:
    TRAIN_DATA_FILE = 'train_data_'+str(IMG_SIZE)+'.npy'
else:
    TRAIN_DATA_FILE = os.path.splitext(args.train_data_file)[0]+'.npy'
if args.test_data_file is None:
    TEST_DATA_FILE = 'test_data_'+str(IMG_SIZE)+'.npy'
else:
    TRAIN_DATA_FILE = os.path.splitext(args.test_data_file)[0]+'.npy'

# Model file
if args.model_file is None:
    MODEL_NAME = 'dogs-vs-cats-convnet_'+str(IMG_SIZE)
else:
    MODEL_NAME = args.model_file

# Function to remove cached data
def rmFile(f):
    try:
        os.remove(f)
    except FileNotFoundError:
        pass

# Maybe delete cached data
if args.clean_images:
    rmFile(os.path.join(TRAIN_DIR, TRAIN_DATA_FILE))
    rmFile(os.path.join(TEST_DIR, TEST_DATA_FILE))
if args.clean_model:
    for file in os.listdir(os.getcwd()):
        if file.find(MODEL_NAME) >= 0:
            rmFile(file)

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
    ret = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
    if len(ret.shape) > 2 and ret.shape[2] == 3:
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    return np.array(ret).reshape(IMG_SIZE, IMG_SIZE, 1)

# parallel executor function, loads an image as gray
def load_img(imgPath):
    img_data = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    if img_data is None:
        return
    return fmt_img(img_data)

def create_data(strDir, strFile, bTrain):
    # If the file exists, return it
    strFullPath = os.path.join(strDir, strFile)
    if os.path.isfile(strFullPath):
        print('Using file on disk', strFullPath)
        return np.load(strFullPath)

    # create new data
    data = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
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

def create_net():
    # Create layer graph
    # Each block here creates a new layer in the network
    tf.reset_default_graph()

    # Input image is a IMG_SIZE x IMG_SIZE gray image
    convnet = input_data(shape = [None, IMG_SIZE, IMG_SIZE, 1], name = 'input')

    # L1
    # max_pool is kind of like a downsampling operation
    # we do this as we progress through the layers and get smaller images
    # 'relu' is a relative unity activation - negative values go to zero
    convnet = conv_2d(convnet, NUM_FILTERS, FILTER_SIZE, activation = 'relu')
    convnet = max_pool_2d(convnet, FILTER_SIZE)

    # L2, twice the number of filters and max_pool
    convnet = conv_2d(convnet, 2 * NUM_FILTERS, FILTER_SIZE, activation = 'relu')
    convnet = max_pool_2d(convnet, FILTER_SIZE)

    # L3, another doubling of filter count
    convnet = conv_2d(convnet, 4 * NUM_FILTERS, FILTER_SIZE, activation = 'relu')
    convnet = max_pool_2d(convnet, FILTER_SIZE)

    # Layers 4 and 5 seem to 'wind back' the filter count
    convnet = conv_2d(convnet, 2 * NUM_FILTERS, FILTER_SIZE, activation = 'relu')
    convnet = max_pool_2d(convnet, FILTER_SIZE)
    convnet = conv_2d(convnet, NUM_FILTERS, FILTER_SIZE, activation = 'relu')
    convnet = max_pool_2d(convnet, FILTER_SIZE)

    # A fully connected layer, which means every "neuron" (or pixel) 
    # in this layer is "connected" to every neuron in the previous layer
    # I think that means we're done downsampling and ready to evaluate
    # what the pixels in this layer's image are telling us
    convnet = fully_connected(convnet, NUM_NEURONS, activation = 'relu')

    # The dropout operation takes the previous layer and scales
    # it by 1/KEEP_THRESH - anything below KEEP_THRESH will
    # shrink - I'm not sure if things actually go to zero...
    KEEP_THRESH = 0.8
    convnet = dropout(convnet, KEEP_THRESH)

    # Create another fully connected layer - I'm not sure why it's softmax
    convnet = fully_connected(convnet, 2, activation='softmax')

    # The regression step is a gradient descent - I think the LR is the
    # step size, and I assume it gets followed until something goes to zero?
    LR = 1e-3
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    return convnet

train_data = create_data(TRAIN_DIR, TRAIN_DATA_FILE, True)

if len(train_data) < args.num_validate:
    raise RuntimeError('Error: not enough validation files!')

# before we can run on the real data, we break off
# a piece of our traning data (the last 500 elements)
# to use as a validation test (training on up till last 500)
model_train = train_data[:-args.num_validate]
model_test = train_data[-args.num_validate:]

# X is the image, Y is whether or not it's a cat or a dog
# The reshape command here takes the array of all images and flattens them
# into one large 4-D array containing images (IMG_SIZE, IMG_SIZE, 1)
X_train = np.array([i[0] for i in model_train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_train = np.array([i[1] for i in model_train])
X_test = np.array([i[0] for i in model_test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_test = np.array([i[1] for i in model_test])

# Try loading model, or retrain if needed
try:
    # Construct the neural network (what was it before?)
    convnet = create_net()
    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
    # see if we can load the model from disk
    model.load(MODEL_NAME)
except tf.errors.NotFoundError:
    # It seems like that error ends the tensorflow session, so
    # reconstruct the net and model to start a new session and learn
    convnet = create_net()
    model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
    model.fit(
        {'input': X_train},
        {'targets': Y_train},
        n_epoch=10,
        validation_set = (
            {'input': X_test},
            {'targets': Y_test}), 
        snapshot_step=500,
        show_metric=True,
        run_id=MODEL_NAME)
    # Save the learned model
    model.save(MODEL_NAME)

# if we are using the camera then open it and wait for input
# press Q to close
if args.cam:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('Camera successfully opened')

        # If there is motion, record a list of frames
        bMotion = False
        lastFrame = None
        dilKern = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        while True:
            # Read a frame, convert frame to gray
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect motion
            bMotionDetected = False
            blur = cv2.GaussianBlur(frame, (21,21), 0)

            # Store previous frame if we don't yet have one
            if lastFrame is None:
                lastFrame = blur
                continue

            # Detect diff betwen frame and find contours
            deltaFrame = cv2.absdiff(lastFrame, blur)
            lastFrame = blur
            ret, thresh = cv2.threshold(deltaFrame, 25, 255, cv2.THRESH_BINARY)
            dilate = cv2.dilate(thresh, None, iterations = 2)
            _, contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > args.min_detect:
                    # draw a rectangle around the contour
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # If we detect motion we start recording
                    # this refreshes time and enters branch below
                    tRecStart = time.time()
                    bMotion = True
                    tDur = 40.
                    break
            
            if bMotion:
                # Stop recording when we hit dur
                # (maybe refresh if motion still detected)
                if time.time() < tRecStart + tDur:
                    bMotion = False
                    bMotionDone = True
                else:
                    # Our operations on the frame come here
                    img = fmt_img(frame)
                    
                    # Run prediction model on image
                    prediction = model.predict([img])[0]

                    # Find the frame with the strongest
                    # dog or cat possibility and store it

            # Show frame
            cv2.imshow('Camera', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cap.release()
                exit()
    else:
        print('Unable to open camera, defaulting to test data set')

# Let's try it out on some test data
test_data = create_data(TEST_DIR, TEST_DATA_FILE, False)
fig=plt.figure(figsize=(16, 12))

# Pick 16 test images at random and see if they're cats or dogs
for num in range(16):
    data = random.choice(test_data)
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4, 4, num+1)
    orig = img_data.reshape(IMG_SIZE, IMG_SIZE)
    # data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([img_data])[0]
    
    if np.argmax(model_out) == 1: 
        str_label='Dog'
    else:
        str_label='Cat'
        
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()