# rAInger
# An autonomous wildlife camera
# For now, however, it's a glorified cat - dog detector
# This is because my training set 

# builtins
import time
import os         
import random
from collections import namedtuple

# OpenCV and numpy
import cv2
import numpy as np

# PIL to read my cat GIF
import PIL

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

# Our util functions
import rAInger_Util as rAI

# get arguments
# These arguments control just about every 
# configurable parameter that gets used here
args = rAI.get_arguments()

# Training and test data file storage\
# If it isn't specified, construct a name for it
if args.train_data_file is None:
    args.train_data_file = 'train_data_'+str(args.img_size)+'.npy'
else:
    args.train_data_file = os.path.splitext(args.train_data_file)[0]+'.npy'
if args.test_data_file is None:
    args.test_data_file = 'test_data_'+str(args.img_size)+'.npy'
else:
    args.test_data_file = os.path.splitext(args.test_data_file)[0]+'.npy'

# Model file
if args.model_file is None:
    args.model_file = 'rAInger_Model_'+str(args.img_size)
else:
    args.model_file = args.model_file
    
# Maybe delete cached data
if args.clean_images:
    rAI.rmFile(os.path.join(args.train_dir, args.train_data_file))
    rAI.rmFile(os.path.join(args.test_dir, args.test_data_file))
if args.clean_model:
    if args.model_file in set(os.listdir(os.getcwd())):
        rAI.rmFile(args.model_file)

# Try loading a cached model from disk
try:
    # The argument of create_net determines if the network is for learning or training
    model = tflearn.DNN(rAI.create_net(False), tensorboard_dir='log', tensorboard_verbose=0)
    model.load(args.model_file)
    print('Using cached network model', args.model_file)

# If it wasn't there, catch the error and train
except tf.errors.NotFoundError:
    print('Unable to load cached network model', args.model_file, ', retraining...')

    # create training data from images
    train_data = rAI.create_data(args.train_dir, args.train_data_file, True)
    if len(train_data) < args.num_validate:
        raise RuntimeError('Error: not enough validation files!')

    # Break apart the data to create a validation set
    model_train = train_data[:-args.num_validate]
    model_test = train_data[-args.num_validate:]

    # X is the image, Y is whether or not it's a cat or a dog
    # The reshape command here takes the array of all images and flattens them
    # into one large 4-D array containing images (IMG_SIZE, IMG_SIZE, 1)
    X_train = np.array([i[0] for i in model_train]).reshape(-1, args.img_size, args.img_size, 1)
    Y_train = np.array([i[1] for i in model_train])
    X_test = np.array([i[0] for i in model_test]).reshape(-1, args.img_size, args.img_size, 1)
    Y_test = np.array([i[1] for i in model_test])

    # It seems like the NotFoundError ends the tensorflow session, so
    # reconstruct the net and model to start a new session
    model = tflearn.DNN(rAI.create_net(True), tensorboard_dir='log', tensorboard_verbose=0)
    
    # Train the model 
    model.fit(
        {'input': X_train},
        {'targets': Y_train},
        n_epoch=args.n_epoch,
        validation_set = (
            {'input': X_test},
            {'targets': Y_test}), 
        snapshot_step=500,
        show_metric=True,
        run_id=args.model_file)

    # Save the learned model
    model.save(args.model_file)

    # Assign the model's net to a prediction net, rather than a learn net
    model.net = rAI.create_net(False)

# If we have a gif argument, conver tall frames to a list of cvImages
imgStream = None
try:
    if args.gif is not None:
        imgStream = rAI.cvGIFSrc(args.gif)
    elif args.cam:
        imgStream = rAI.cvVideoSrc(0)
except (IOError, RuntimeError) as e:
    print('Unable to open video source')
    imgStream = None

# if we are using the camera then open it and wait for input
# press Q to close
if imgStream is not None:
    # If we detect motion, we will record a series of frames 
    # and find the frame with the strongest prediction. We'll
    # cache these and send the best predictions over LoRaWAN
    if args.motion:
        tRecStart = 0.
        motionDetector = rAI.cvMotionDetector(args.min_detect)
        BestPrediction = namedtuple('BestPrediction', {'strength', 'index', 'image'})
        bestPrediction = None
        liBestPredictions = []

    # If we don't care about motion detection, always do prediction
    bDoPrediction = not args.motion

    # Read frames from the camera
    hasFrames = True
    while hasFrames:
        hasFrames, frame = imgStream.getFrame()
        if not hasFrames:
            break

        # Maybe detect motion
        if args.motion:
            # If we detect motion, cache current time
            # This will start or refresh record timer
            if motionDetector.detect(frame, True):
                tRecStart = time.time()
                bDoPrediction = True
            # If no motion is detected and we exceed the timer, turn off prediction and
            # add best prediction we got during this period to the list of detected objects
            elif bDoPrediction and time.time() > (tRecStart + args.motion_window):
                if bestPrediction:
                    liBestPredictions.append(bestPrediction)
                    bestPrediction = None
                bDoPrediction = False
                print('No motion detected')

        # If we are doing prediction
        if bDoPrediction:
            # Run prediction model on image
            modelInput = [rAI.fmt_img(frame)]
            prediction = model.predict(modelInput)[0]
            if args.motion:
                # bestPrediction is a tuple(predictionVal, maxIdx, frame)
                # The maxIdx will be 0 for cats and 1 for dogs. Assign it
                # if it's our first or take the one with the stronger max element
                if not bestPrediction:
                    bestPrediction = BestPrediction(prediction, np.argmax(prediction), frame)
                elif np.max(bestPrediction.strength) < np.max(prediction[0]):
                    bestPrediction = BestPrediction(prediction, np.argmax(prediction), frame)
                    # print('best', np.max(bestPrediction.strength))

            # What do we think is in this frame?
            if prediction[np.argmax(prediction)] > args.thresh:
                print('I think it\'s a', ['cat', 'dog'][np.argmax(prediction)])
            else:
                print('Doesn\'t look like anything to me...')
            # print(prediction)

        # Show frame
        cv2.imshow('Predicting...', frame)

        # Advance frame by frame if in GIF mode
        key = cv2.waitKey(0 if args.gif else 1)

        # Quit if 'q' is pressed
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # If we broke out and there was one more
    # "best prediction", add it too
    if bestPrediction:
        liBestPredictions.append(bestPrediction)

    # For every "best prediction" we had, send its data over the antenna
    # (the index member is the max arg, 0 for cat and 1 for dog)
    detectBytes = bytearray([bp.index for bp in liBestPredictions])
    rAI.send_lora_data(detectBytes)

# Run the code on some test data, if it exists, and plot the output
# This is taken from the Medium example, and is a useful network validator
else:
    test_data = rAI.create_data(args.test_dir, args.test_data_file, False)
    fig = plt.figure(figsize=(16, 12))
    
    # Pick 16 test images at random and see if they're cats or dogs
    setChosen = set()
    for num in range(16):
        # Make sure the images are unique
        ixRnd = random.randint(0, len(test_data)) 
        while ixRnd in setChosen:
            ixRnd = random.randint(0, len(test_data)) 
        setChosen.add(ixRnd)
    
        # Pull test data and unpack
        data = random.choice(test_data)
        img_num = data[1]
        img_data = data[0]
        
        # Plot image
        y = fig.add_subplot(4, 4, num+1)
        orig = img_data.reshape(args.img_size, args.img_size)
        y.imshow(orig, cmap='gray')
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    
        # Label with prediction
        model_out = model.predict([img_data])[0]
        if np.argmax(model_out) == 1: 
            str_label='Dog'
        else:
            str_label='Cat'
        plt.title(str_label)
    
    # Show plot
    plt.show()