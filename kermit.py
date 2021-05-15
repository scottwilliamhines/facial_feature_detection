
from flask import Flask, render_template, Response
import numpy as np
import cv2
import dlib


app = Flask(__name__)

capture = cv2.VideoCapture(2)
face_detector = dlib.get_frontal_face_detector()
feature_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def feature_landmarks():
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

            for n in range(0,68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                cv2.circle(img=frame, center=(x,y), radius=8, color= (0,255,0), thickness=-1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def kermit_face():
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
            vector = np.empty([68, 2], dtype = 'float32')
            for b in range(68):
                vector[b][0] = landmarks.part(b).x
                vector[b][1] = landmarks.part(b).y

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

            src_pts_1 =  np.genfromtxt('kermit_labels.csv', delimiter=',')
            src_pts = np.empty([37,2], dtype = 'float32')
            for i, row in enumerate(src_pts_1):
                src_pts[i][0] = row[1]
                src_pts[i][1] = row[2]

            # load mask image
            mask_img = cv2.imread('kermit.png', cv2.IMREAD_UNCHANGED)
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

            # mask overlay
            alpha_mask = transformed_mask[:, :, 3]
            alpha_image = 1.0 - alpha_mask

            mask = (transformed_mask[:,:,3] != 0).flatten()
            kermit = transformed_mask[:,:,0:3].flatten().reshape(-1,3)
            cap = frame.flatten().reshape(-1,3)
            cap[mask] = kermit[mask]
            final = cap.reshape(frame.shape[0],frame.shape[1],3)
    
            ret, buffer = cv2.imencode('.jpg', final)
            final = buffer.tobytes()
            yield (b'--final\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n')
            
def miss_piggy_face():
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
            vector = np.empty([68, 2], dtype = 'float32')
            for b in range(68):
                vector[b][0] = landmarks.part(b).x
                vector[b][1] = landmarks.part(b).y

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

            src_pts_1 =  np.genfromtxt('miss_piggy_labels_2.csv', delimiter=',')
            src_pts = np.empty([37,2], dtype = 'float32')
            for i, row in enumerate(src_pts_1):
                src_pts[i][0] = row[1]
                src_pts[i][1] = row[2]

            # load mask image
            mask_img = cv2.imread('miss_piggy.png', cv2.IMREAD_UNCHANGED)
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

            # mask overlay
            alpha_mask = transformed_mask[:, :, 3]
            alpha_image = 1.0 - alpha_mask

            mask = (transformed_mask[:,:,3] != 0).flatten()
            piggy = transformed_mask[:,:,0:3].flatten().reshape(-1,3)
            cap = frame.flatten().reshape(-1,3)
            cap[mask] = piggy[mask]
            final = cap.reshape(frame.shape[0],frame.shape[1],3)
    
            ret, buffer = cv2.imencode('.jpg', final)
            final = buffer.tobytes()
            yield (b'--final\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n')
            
def gonzo_face():
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
            vector = np.empty([68, 2], dtype = 'float32')
            for b in range(68):
                vector[b][0] = landmarks.part(b).x
                vector[b][1] = landmarks.part(b).y

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

            src_pts_1 =  np.genfromtxt('gonzo_labels.csv', delimiter=',')
            src_pts = np.empty([37,2], dtype = 'float32')
            for i, row in enumerate(src_pts_1):
                src_pts[i][0] = row[1]
                src_pts[i][1] = row[2]

            # load mask image
            mask_img = cv2.imread('gonzo.png', cv2.IMREAD_UNCHANGED)
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

            # mask overlay
            alpha_mask = transformed_mask[:, :, 3]
            alpha_image = 1.0 - alpha_mask

            mask = (transformed_mask[:,:,3] != 0).flatten()
            piggy = transformed_mask[:,:,0:3].flatten().reshape(-1,3)
            cap = frame.flatten().reshape(-1,3)
            cap[mask] = piggy[mask]
            final = cap.reshape(frame.shape[0],frame.shape[1],3)
    
            ret, buffer = cv2.imencode('.jpg', final)
            final = buffer.tobytes()
            yield (b'--final\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + final + b'\r\n')

@app.route('/kermit')
def kermit():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(kermit_face(), mimetype='multipart/x-mixed-replace; boundary=final')

@app.route('/kermit_window')
def kermit_window():
    return render_template('kermit.html')

@app.route('/miss_piggy')
def miss_piggy():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(miss_piggy_face(), mimetype='multipart/x-mixed-replace; boundary=final')

@app.route('/miss_piggy_window')
def miss_piggy_window():
    return render_template('miss_piggy.html')

@app.route('/gonzo')
def gonzo():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gonzo_face(), mimetype='multipart/x-mixed-replace; boundary=final')

@app.route('/gonzo_window')
def gonzo_window():
    return render_template('gonzo.html')

@app.route('/landmark_vis')
def landmark_vis():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(feature_landmarks(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/landmark_window')
def landmark_window():
    return render_template('landmark.html')

@app.route('/')
def index():
    display_str = 'Hello! Pick a Muppet'
    # code below makes a button to go to the '/rainbow' block
    go_to_muppet_html = '''
        <br>
        <form action="/landmark_window" >
            <input type="submit" value = "Landmarks"/>
        </form>
        <br>
        <form action="/kermit_window" >
            <input type="submit" value = "Kermit"/>
        </form>
        <br>
        <form action="/miss_piggy_window" >
            <input type="submit" value = "Miss Piggy"/>
        </form>
        <br>
        <form action="/gonzo_window" >
            <input type="submit" value = "Gonzo"/>
        </form>
        <br> 
    '''
    # html that gets returned
    return display_str + go_to_muppet_html


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
