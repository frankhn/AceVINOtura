# AceVINOtura

AceVINOtura is an application that detects bad cats that are used to jump on forbidden areas such as kitchen counter, dining table and more. the application can be used to detect all of these bad cats as well as differentiate them from good one.

## About the AceVINOtura

![](images/vid7_goodcat.png?raw=true) ![](images/vid7_badcat.png?raw=true)


## Prerequisites

You need Intel's OpenVino toolkit down installed on your machine and the OpenCV as well.
You can read more about both these.

OpenCV    (<https://opencv.org>) 
OpenVino toolKit (<https://software.intel.com/en-us/openvino-toolkit>)


## Getting Started

Firstly you'll need to clone the repo down to you local machine.

``` git clone https://github.com/frankhn/AceVINOtura.git ```

Once you have the project.

``` python acevinotura.py -i videos\vid7.mp4 -z "[[0, 100], [470, 160], [260, 350], [0, 290]]" ```

## Licence

Distributed under the MIT License. See LICENSE for more information.

## Acknowledgements

. https://github.com/frankhn

. https://github.com/JulienGremillot

. https://github.com/JamesDBartlett

. https://github.com/theadnanerfan
