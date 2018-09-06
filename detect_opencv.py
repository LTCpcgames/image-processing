import cv2,io,imutils
import numpy
from imutils.video import VideoStream


face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')

def display_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    '''
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('Screen',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def use_webcam():
    print("Starting Web Cam...........")
    cap = cv2.VideoCapture(0)
    while(True):
        ret,img = cap.read()
        display_face(img)
        
    
    
def use_picamera():
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    # Init the camera and grab a ref to the raw camera capture
    print("Starting Pi Camera...........")
    camera = PiCamera()
    #camera = cv2.VideoCapture(0)
    camera.resolution = (640, 480)
    #camera.framerate = 32
    rawCapture = PiRGBArray(camera,(640,480))

    # Allow Camera to Warm up
    time.sleep(0.1)

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        display_face(image)
        rawCapture.truncate(0)
        rawCapture.seek(0)


#use_webcam()
use_picamera()


cap.release()
cv2.destroyAllWindows()
