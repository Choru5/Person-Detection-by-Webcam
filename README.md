# Person/Pet-Detection-by-Webcam

Requirements:
 - A PC with a Nvidia GPU
 - A smartphone / webcam
 
 What it is:
 -  This is a machine learning model that has been trained to detect people / pets and notify you.
 
 Installation Instructions:
 
 - Download this repository.
 - Make sure you have Python 3.8 installed
 - Install anaconda (https://www.anaconda.com/products/distribution) on your computer.
 - Open anaconda as administrator and cd to the directory of the persondetector.py script in this repository.
 - Install pytorch and cuda (https://pytorch.org/get-started)
 - Create a virtual environment using this command ``` conda create -n pytorch-gpu python==3.8 ```
 - Activate the virtual env using ``` conda activate pytorch-gpu ```
 - Install required packages by running ``` pip install -r requirements.txt ```
 - Download Iriun on your PC and the Iriun app on your phone (if using a webcam ignore this)

 Customizable Settings:
 ```
 visuals = True  # This will toggle the visiblity of the camera feed
 
 IN = 1  # This will toggle the webcam source (0 = normal webcam, 1 = smartphone DroidCam webcam)
 
 target = 'person'  # This changes what you are detecting, examples are 'cat', 'dog', or 'person' by default 
 
 modelFile = "yolov5s.pt"  # This is the AI model file, do not change this unless you really know what you are doing!
```

 How to use:
 
 - Open anaconda as administrator and cd to the directory of the persondetector.py script in this repository.
 - Run the command ``` conda activate pytorch-gpu ``` to start up the virtual environment
 - Run the command ``` python pythondetector.py ``` to start the program, make sure you have Iriun open on your PC + phone and are connected to the same Wi-FI network as your phone (if using a webcam ignore this)
