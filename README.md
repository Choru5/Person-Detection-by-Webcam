# Person-Detection-by-Webcam

Requirements:
 - A Nvidia GPU
 
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
 - Download DroidCam Client on your PC and the DroidCam app on your phone

 Customizable Settings:
 ```
 visuals = True  # This will toggle the visiblity of the camera feed
 
 IN = 1  # This will toggle the webcam source (0 = normal webcam, 1 = DroidCam webcam)
 
 target = 'person'  # This changes what you are detecting, examples are 'cat', 'dog', or 'person' by default 
 
 modelFile = "yolov5s.pt"  # This is the AI model file, do not change this unless you really know what you are doing!
```

 How to use:
 
 - Open anaconda as administrator and cd to the directory of the persondetector.py script in this repository.
 - Run the command ``` conda activate pytorch-gpu ``` to start up the virtual environment
 - Run the command ``` python pythondetector.py ``` to start the program, make sure you have DroidCam Client open and are connected to your phones DroidCam via Wi-Fi or USB
