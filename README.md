<!--
<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://raw.githubusercontent.com/amitmerchant1990/electron-markdownify/master/img/markdownify.png" alt="Markdownify" width="200"></a>
  <br>
  Markdownify
  <br>
<img src="https://media.giphy.com/media/8vQSQ3cNXuDGo/giphy.gif" width="400" height="400" />
</h1>
-->

# rAInger
An autonomous wildlife camera <br>
(<a href="https://challengerocket.com/nvidia">NVIDIA® Jetson™ Developer Challenge</a> submission)
<br><br>

The goal of this project is to design a program that enables an Nvidia Jetson computer equipped with a camera to autonomously capture and analyze pictures.  These pictures are fed to a convolutional neural network (CNN) trained to recognized various types of animals, allowing the Jetson to independently track wildlife in the field. 

That's the theory, anyway. I don't have easy access to wildlife training set, so my networks are trained on cats and dogs. The high level purpose and design of this program is described further this short video, which I've included as part of my 
submission. Rather than repeat that information, I'll use this README to describe the actual implementation. 

<br>
**0. File Index**
<br>
There are three source files in this repository
<br>
- rAInger.py - the main script, which runs the code that trains the network and uses it to analyze images
- rAInger_Util.py - an auxilliary script where most of the functions used in rAInger.py are implemented
- rAInger_pyl.cpp - A C++ source file that communicates with an onboard radio to transmit analysis data

Additionally there are two subrepositories (<a href="https://github.com/mirakonta/lora_gateway">lora\_gateway</a> and <a href="https://github.com/mynameisjohn/PyLiaison">PyLiaison</a>). The former is used for radio communication, and the latter allows the python code to communicate with C++ code. 

<br>
**1. Training**
<br>
The first thing we have to do is train our CNN. This is done with Python using the following libraries:
- numpy (array and math operations)
- opencv (for image IO and processing)
- tensorflow-gpu (underlying network and CUDA implementation)
- tflearn (high level DNN that wraps tensorflow code)
- matplotlib (optional, used for testing and verification)

All of these were readily available to me on pip, so getting this code up and running should be straightforward. The data used to train the network was obtained from <a href="https://www.kaggle.com/c/dogs-vs-cats/data">kaggle</a> as a part of their Dogs vs. Cats challenge. 
<br>

Because the training step is computationally intense I chose to run it on my desktop computer which has an Nvidia 980 Ti graphics card. Once the network was trained I was able to save it to a file and load it pre-trained on the Jetson. 

<br>
**2. Prediction**
<br>
Once the network is trained we can use it to analyze the content of images. Because it was trained on cats and dogs that's what it can recognize, but with a more diverse training set it could be used to analyze images of other types of wildlife (or anything, really. )
<br>

We run the prediction code on live input. The code uses a simple OpenCV motion detector to wait for something interesting to go by, and when it does it runs the prediction code to determine what it's seeing. 
<br>

In the field the input will be coming from a camera, but for testing purposes I optionally allow reading a <a href="https://giphy.com/gifs/cat-moment-remember-8vQSQ3cNXuDGo">.GIF file</a> as input. 

<img src="https://media.giphy.com/media/1wpxEWMljk7cv6nA96/giphy.gif" width="400" height="400"/>

The rectangles you see are the areas in which motion was detected. We take motion as a trigger to start running images through the prediction network - while motion occurs we find the image with the strongest (in this case, which image is the most "cat or dog"-like). That gets us the image below:

<img src="https://i.imgur.com/pNTmU5Y.png" width="400" height="400"/>
[1.3075967,  0.       ]

Our network tells us that it's a cat (it must be that little nose!), so we send a message over our radio indicating as such to a central database. 

