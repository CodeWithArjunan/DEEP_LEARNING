from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound

frequency =2500
durtion = 2000

def eyeAspectRatio(eye):
    a = dist.euclidean(eye[1],eye[5])
    b = dist.euclidean(eye[2],eye[4])
    c = dist.euclidean(eye[0],eye[3])

    ear = (a+b)/(2.0*c)

    return ear

count = 0
earThresh = 0.3
earFrames = 48
shapePredictor = 'shape_predictor_68_face_detector.dat'

cam = cv2.Videocapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

(lstart,Lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(Rstart,Rend) = face.utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _,frame = cam.read(0)
    frame = imutils.resize(frame,width=450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)

    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[Lstart:Lend]
        rightEye = shape[Rstart:Rend]

        leftEar = eyeAspectRatio(leftEye)
        rightEar = eyeAspectRatio(rightEye)

        ear = (leftEar + rightEar)/2.0

        leftEyeHull = cv2.convexHull(leftEye)
        righttEyeHull = cv2.convexHull(rightEye)
        cv2.drowContours(frame,[leftEyeHull],-1,(0,0,255),5)
        cv2.drowContours(frame,[rightEyeHull],-1,(0,0,255),5)

        if ear < earThresh:
            count+=1
            if count>=earFrames:
                cv2.putText(frame,"Drowsiness detection",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                windsound.Beep(frequency,duration)
        else:
            count=0

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindow()
