import numpy as np
import cv2
#import time
import imutils

prototxt='MobileNetSSD_deploy.prototxt.txt'
model='MobileNetSSD_deploy.caffemodel'
confthresh =0.2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","mobile"]

colors = np.random.uniform(0,255,size=(len(CLASSES),3))
print('Loading model')

net = cv2.dnn.readNetFromCaffe(prototxt,model)
print('Model loaded')
print('load camera....')

cam =cv2.VideoCapture(0)

while True:
    _,img =cam.read()
    img=imutils.resize(img,width=1000)
    (h,w)=img.shape[:2]

    imgResizeBlob = cv2.resize(img,(300,300))
    blob = cv2.dnn.blobFromImage(imgResizeBlob,0.007843,(300,300),12.5)

    net.setInput(blob)
    detection = net.forward()
    #print('detection')

    detshape = detection.shape[2]
    
    for i in np.arange(0,detshape):
        confidance = detection[0,0,i,2]
        if confidance > confthresh:
            idx = int(detection[0,0,i,1])
            #print('ID:',detection[0,0,i,1])

            box = detection[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            lable = "{}:{:.2f}%".format(CLASSES[idx],confidance*100)

            cv2.rectangle(img,(startX,startY),(endX,endY),colors[idx],2)
'''
            if start€y-15 > 15:
                startY-15
            else:
                startY+15
'''
            cv2.putText(img,lable,(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,0.5,colors[idx],2)



    cv2.imshow("Frame",img)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()


        
