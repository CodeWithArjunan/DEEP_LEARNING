import cv2
import imutils

CaseCade_src = 'cars.xml'

car_caseCade = cv2.CascadeClassifier(CaseCade_src)

cam = cv2.VideoCapture(0)

while True:
    _,img = cam.read()
    img = imutils.resize(img,width=500)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cars = car_caseCade.detectMultiScale(gray,1.1,1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Frame",img)

    b=str(len(cars))
    a=int(b)
    n=a
    print('-------------------------------------------')
    print("North : %d "%n)
    if(n>8):
        print("North More Traffic, Please on the RED signal")
    else:
        print("No Traffic")
    
    if cv2.waitKey(33)==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()



