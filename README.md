# AceVINOtura

AceVINOtura is an application that detects bad cats that are used to jump on forbidden areas such as kitchen counter, dining table and more. the application can be used to detect all of these bad cats as well as differentiate them from good one.

## About the AceVINOtura

![](images/vid7_goodcat.png?raw=true) ![](images/vid7_badcat.png?raw=true)


## Prerequisites

You need Intel's OpenVino toolkit down installed on your machine and the OpenCV as well.
You can read more about both these.

- OpenCV    (<https://opencv.org>) - The simple install should look like ```pip install opencv-python```. 
- OpenVino toolKit (<https://software.intel.com/en-us/openvino-toolkit>) - See website for installation depending of your configuration.
- (optional) TQDM (<https://github.com/tqdm/tqdm>) - Only for use on local files, install with ```pip install tqdm```.
- (optional) FFMPEG & H264 codec - On raspberry/linux : ```sudo apt-get install ffmpeg x264 libx264-dev```

## Getting Started

Firstly you'll need to clone the repo down to you local machine.

``` git clone https://github.com/frankhn/AceVINOtura.git ```

Once you have the project, you can test that the script is working with a local video file (test file provided).

``` python acevinotura.py -i videos\vid7.mp4 -z "[[0, 100], [470, 160], [260, 350], [0, 290]]" ```

To install it on a raspberry pi, you'll have to assure that your picam module is fixed and facing the zone to watch.
Then take a picture from the chosen location with the command :

``` raspistill  -w 640 -h 480 -o test.jpg ```

With this picture, you'll be able to use a software like [The Gimp](https://www.gimp.org/) to note the points delimiting the forbidden zone to watch.

For example, I've extracted 4 points : [[365, 435], [380, 390], [500, 390], [540, 435]] (corresponding to the corner of my table)

Then you can start detection with :

``` python acevinotura.py -z "[[365, 435], [380, 390], [500, 390], [540, 435]]" -s True -d MYRIAD -o out```

(Here I've used the "-s True" parameters to trace the forbidden zone, and the "-o out" parameters to save the detected images in a "out" directory)

## Licence

Distributed under the MIT License. See LICENSE for more information.

## Acknowledgements

. https://github.com/frankhn

. https://github.com/JulienGremillot

. https://github.com/JamesDBartlett

. https://github.com/theadnanerfan
