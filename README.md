# Facial Feature Detection
Active facial feature detection and it's uses in the world

### Project Goals
My goal with this project was to create a Snapchat-like filter that is reproduceable on live video capture. We do this by using a machine learning model to actively predict certain facial feature landmarks frame by frame as video is being captured by the camera. Because it is work on a frame by frame basis, and most computer cameras operate at roughly 30 frames per second, the model has perform these predictions really quickly. For my purposes I decided to go with the DLIB library to help with the facial feature predictions. 

### Dependancies
If you have an interest in re-creating or running the code in this repo, then you will want to install the following dependencies to your Python environment.

[DLIB](http://dlib.net/)

[NUMPY](https://numpy.org/doc/)

[CV2](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

**To build the web application:**

[Flask](https://flask.palletsprojects.com/en/2.0.x/)

### The Process
I was fortunate in planning for this project to stumble upon this [Article](https://towardsdatascience.com/detecting-face-features-with-python-30385aee4a8e) on **Towards Data Science** written by Juan Cruz Martinez. He pointed me towards the DLIB library and outlines a very effective method for getting a facial feature detector up and running on CV2. The process is as follows:

1. Import your dependencies:
   
    import cv2
    import dlib
    import numpy as np
    import os

2. Use DLIB to detect the face and draw a bounding box over each face in the image.
   
    face_detector = dlib.get_frontal_face_detector()
    
    