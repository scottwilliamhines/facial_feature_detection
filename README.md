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

![](https://github.com/scottwilliamhines/facial_feature_detection/blob/main/readme_assets/facial_detection_gif.gif)

---

### A quick note on how DLIB's shape predictor works:

DLIB's face predictor passes a sparse matrix of pixel information from the grayscale `frontal_face_detector` as features into an ensemble of gradient boosted trees model predicting on the (x,y) coordinates of our landmarks. The high level idea of how this trains is as follows:

- it takes the average "Shape" (which is a vector of the x,y coordinates of all our landmarks) and passes those in as a prediction for the shape of every image in the training set. 

- It then calculates the residuals from those predictions. 

- The next step is to create an ensemble of gradient boosted trees and predict on the residuals based on the features selected. 

- We multiply these predictions by the learning rate and add that to our original predicted x,y coordinates. 

- Once we have our slightly adjusted predicted coordinates, we can update our residuals and start the process all over again by predicting new residuals based on features, averaging the values in each individual leaf, multiplying by our learning rate, and updating our coordinates again with the residuals of this new tree. 

- This process continues until we reach a pre-set number of trees or the residuals are no longer changing our predicted coordinates by a significant amount. 

---

### Adding an Overlay

Now that we have the code all put together to predict the landmarks, we can begin adding overlay images that track to those landmarks. To do this we need to annotate any images that we want to overlay with the coordinates of the landmarks we want the image to track to. For this purpose I used an online application called [makesense](https://www.makesense.ai/). 

- Select get started

- Add labels coorespnding to the numbers of the landmarks as seen below

<img src="https://github.com/scottwilliamhines/facial_feature_detection/blob/main/readme_assets/facial_landmarks.jpeg"
     width = "300"/>
     
- Then add points on the image and add the correct labels to them. 

- Finally, export a csv with the annotations. 

With these annotations we can put all of the pieces together and get a working filter with the following steps.

1. Repeat steps 1 thru 4 above giving us the following code. 

        face_detector = dlib.get_frontal_face_detector()
        feature_predictor = dlib.shape_predictor('model_assets/shape_predictor_68_face_landmarks.dat')
        
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
                
2. Load an empty numpy array with the x and y coordinates of the landmarks predicted by our model.

                 vector = np.empty([68, 2], dtype = 'float32')
                 for b in range(68):
                     vector[b][0] = landmarks.part(b).x
                     vector[b][1] = landmarks.part(b).y
                     
3. Choose the landmarks we want to use for our overlay

                 dst_pts = np.array(
                    [
                        vector[0],
                        vector[1],
                        vector[2],
                        vector[3],
                        vector[4],
                        vector[5],
                        vector[6],
                        vector[7],
                        vector[8],
                        vector[9],
                        vector[10],
                        vector[11],
                        vector[12],
                        vector[13],
                        vector[14],
                        vector[15],
                        vector[16],
                        vector[49],
                        vector[50],
                        vector[51],
                        vector[52],
                        vector[53],
                        vector[54],
                        vector[55],
                        vector[56],
                        vector[57],
                        vector[58],
                        vector[59],
                        vector[60],
                        vector[61],
                        vector[62],
                        vector[62],
                        vector[63],
                        vector[64],
                        vector[65],
                        vector[66],
                        vector[67],
                    ])
                    
4. Load a numpy array with the x and y coordinates from our annotations.

                src_pts_1 =  np.genfromtxt('model_assets/kermit_labels.csv', delimiter=',')
                src_pts = np.empty([37,2], dtype = 'float32')
                for i, row in enumerate(src_pts_1):
                    src_pts[i][0] = row[1]
                    src_pts[i][1] = row[2]
                    
5. Load in the overlay image and create a kernel that will transform and re-size the image as it tracks along with the landmarks. 


                mask_img = cv2.imread('model_assets/kermit.png', cv2.IMREAD_UNCHANGED)
                mask_img = mask_img.astype(np.float32)

                M, _ = cv2.findHomography(src_pts, dst_pts)

                transformed_mask = cv2.warpPerspective(
                mask_img,
                M,
                (frame.shape[1], frame.shape[0]),
                None,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
                )
                
6. Make an alpha mask our of the image, apply it to the original captured frame and then replace the masked pixels with the transformed overlay. 

                alpha_mask = transformed_mask[:, :, 3]
                alpha_image = 1.0 - alpha_mask
                trans_h = transformed_mask.shape[0]
                trans_w = transformed_mask.shape[1]

                mask = (transformed_mask[:,:,3] != 0).flatten()
                kermit = transformed_mask[:,:,0:3].flatten().reshape(-1,3)
                cap = frame.flatten().reshape(-1,3)
                cap[mask] = kermit[mask]
                final = cap.reshape(frame.shape[0],frame.shape[1],3)
                 
All that is left after that is to show your final product. We are left with something like this. 

<img src="https://github.com/scottwilliamhines/facial_feature_detection/blob/main/readme_assets/kermit_face.gif"
     width = "300"/>
     
 ### Conclusion
 
 I had a really great time with this project. There are a lot of amazing resources out there for this because it isn't exactly a new problem to solve. However, I think that the use cases for this type of tech a broad and far reaching. This type of feature detection can help with eye tracking statistics, which marketing teams can use to determine the effectiveness of their adds. It can be used to model new products before purchasing them. It can be used in tv and film to assist in creating effects. It can also be used in physical therapy to help determine remotely if a client is performing their exercises correctly. I believe that there is a bright present for this tech and an even brighter future. I have loved learning more about it and I am very much looking forward to continuing my path of discovery. 