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
- rAInger.py - the main script, which runs the code that trains the network and uses it to analyze images <br>
- rAInger_Util.py - an auxilliary script where most of the functions used in rAInger.py are implemented <br>
- rAInger_pyl.cpp - A C++ source file that communicates with an onboard radio to transmit analysis data <br>

Additionally there are two subrepositories (<a href="https://github.com/mirakonta/lora_gateway">lora\_gateway</a> and <a href="https://github.com/mynameisjohn/PyLiaison">PyLiaison</a>). The former is used for radio communication, and the latter allows the python code to communicate with C++ code. 

<br>
**1. Training**
<br>
The first thing we have to do is train our CNN. This is done with Python using the following libraries: <br>
- numpy (array and math operations) <br>
- opencv (for image IO and processing) <br>
- tensorflow-gpu (underlying network and CUDA implementation) <br>
- tflearn (high level DNN that wraps tensorflow code) <br>
- matplotlib (optional, used for testing and verification) <br>

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

The rectangles you see are the areas in which motion was detected. We take motion as a trigger to start running images through the prediction network - while motion occurs we find the image with the strongest (in this case, which image is the most "cat or dog"-like). That strength gets returned to us as a 2-element array - the first element is how certain we are it's a cat, and the second how certain we are it's a dog. We want the image with the maximum element - we don't care if it's the first of the second, just so long as it's the max. That gets us this image:

<img src="https://i.imgur.com/pNTmU5Y.png" width="400" height="400"/>

The vector we're returned is ```[1.3075967,  -1.1696696]```. Adding 'softmax' layer to our network ensures the values sum to 1, giving us something like a percent certainty:  ```[0.9225327  0.07746734]```. Given that this is the strongest prediction we got during the motion window, we can say we're about 92% certain there's a cat in front of us (it must be that itty bitty nose!)

The reason we choose not to use the softmax layer is that we can't always be certain that what triggered the motion is something we recognize. If, for example a hamster were to cross in front of the camera, the softmax layer would still feel certain that it was either a cat. This is useful while training because we know we're feeding it something we'll recognize, but in the wild we want to know if it's something we don't recognize. 

<br>
**3. Communication**
<br>

Now that we've figured we've seen something, we've got to send it over to our "central database". Admittedly this is an abstract concept for the time being, but what we can do is transmit some data over a LoRaWAN antenna to some remote gateway. The easiest way I know of doing this is using the <a href="https://github.com/mirakonta/lora_gateway">lora\_gateway</a> library (libloragw). This kind of thing could be implemented in Python, but I honestly don't trust myself to do it on such short notice. 
<br>

However, libloragw is a C library, and in order for us to use tensorflow and tflearn we've got to be using Python. My solution is to use <a href="https://github.com/mynameisjohn/PyLiaison">PyLiaison</a>, a library I wrote to make it easy to call C++ from Python and vice versa. I use any opportunity I can to plug the library, but in situations like this it's quite useful. 
<br>

What we really need is a way of sending raw bytes over the antenna. We've conveniently given cats an label of 0 and dogs an label of 1, so all we need to do is send a 0 or 1 each time we see a cat or dog, respectively. With this function, 

```C++
// Sends raw bytes over the antenna to our gateway
bool send_loro_data( std::vector<char> vData )
{
	// create packet from template, store packet
	struct lgw_pkt_tx_s txPkt = g_TxPktTemplate;
	txPkt.size = std::min<uint16_t>( vData.size(), sizeof( txPkt.payload ) );
	memcpy( txPkt.payload, vData.data(), txPkt.size );

	// send - this is a non-blocking call so don't bunch up...
	return LGW_HAL_SUCCESS == lgw_send( txPkt );
}
```

We can invoke this function from Python by exposing it into a module like so

```C++
// Create python module named pylLoRaWAN with send_loro_data
pyl::ModuleDef * pLLGWDef = pylCreateMod( pylLoRaWAN );
pylAddFnToMod( pLLGWDef, send_loro_data );
```

and calling it from Python

```Python
def send_lora_data(data):
    if not (isinstance(data, bytes) or isinstance(data, bytearray)):
        raise RuntimeError('We can only send bytes over LoRaWAN')
    return pylLoRaWAN.send_lora_data(data)
```

The data variable, if it is a ```bytes``` or ```bytearray``` type, will be converted to a ```std::vector<char>``` and used by the C++ code in the ```send_loro_data``` function. In this way we can string together a bytearray of what we've seen and send it over the antenna. 
