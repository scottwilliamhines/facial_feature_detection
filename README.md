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

2. Instantiate a variable with DLIB's frontal face detector. This allows us to isolate any faces in the image and draw a bounding box on them. 
   
        face_detector = dlib.get_frontal_face_detector()
    
    
3. Instantiate a variable with DLIB's shape predictor. Here I pass in a pre-trained model that I found at this [github](https://github.com/italojs/facial-landmarks-recognition) by Italo Jose. Additional credit goes to the article linked above by Juan Cruz Martinez for pointing me in the right direction. 

        feature_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
4. In this next bit of code we are able to use `CV2` to capture video from our designated camera, capture frames from that camera, convert each frame to grayscale, pass those frames into our `face_detector` and then pass each face from the face detector into our `shape_predictor`.

        capture = cv2.VideoCapture(2)

        while True:
            _, frame = capture.read()
    
            grayscale = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

            faces = face_detector(grayscale)

            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                landmarks = feature_predictor(image = grayscale, box=face)
                
5. Finally, once we have our landmarks predicted, we can draw a circle on the `frame` for every landmark position. 


                for n in range(0,68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
            
                    cv2.circle(img=frame, center=(x,y), radius=8, color= (0,255,0), thickness=-1)
    
            cv2.imshow(winname = "Facial Feature Detection", mat=frame)
        
            if cv2.waitKey(20) & 0xFF == 27:
            break
    
        capture.release()

        cv2.destroyAllWindows()
        
Below is a short example of what the code above produces on live video. 

![](https://github.com/scottwilliamhines/facial_feature_detection/blob/main/facial_detection_gif.gif)

---

**A quick note on how DLIB's shape predictor works:**

DLIB's face predictor passes a sparse matrix of pixel information from the grayscale `frontal_face_detector` as features into an ensemble of gradient boosted trees model predicting on the (x,y) coordinates of our landmarks. The high level idea of how this trains is that it takes the average "Shape" (which is a vector of the x,y coordinates of all our landmarks) and passes those in as a prediction for the shape of every image in the training set. It then calculates the residuals from those predictions. The next step is to create an ensemble of gradient boosted trees and predict on the residuals based on the features selected. We multiply these predictions by the learning rate and add that to our original predicted x,y coordinates. Once we have our slightly adjusted predicted coordinates, we can update our residuals and start the process all over again by predicting new residuals based on features, averaging any duplicates, multiplying by our learning rate, and updating our coordinates again with the residuals of the new tree. This process continues until we reach a pre-set number of trees or the residuals are no longer changing our predicted coordinates by a significant amount. 

---

### Adding an Overlay

Now that we have the code all put together to predict the landmarks, 
        