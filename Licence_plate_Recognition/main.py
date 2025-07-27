import cv2
import pytesseract 

#Read original img
img=cv2.imread('car1.jpg')
cv2.imshow("original",img)
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#Read gray img
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray_img",gray_img)

#Canny edge detection
canny_edge = cv2.Canny(gray_img,170,200)
cv2.imshow("Canny_img",canny_edge)

#Find contor 
contours,new = cv2.findContours(canny_edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:30]

#Initialize the license plate contour
contour_with_licence_plate = None
license_plete = None
x=None
y=None
w=None
h=None

#contour blanck contours on
contour_img = img.copy()
#Draw contours on the blanck
cv2.drawContours(contour_img,contours,-1,(0,0,255),2)
cv2.imshow("img with countors & Lince plate",contour_img)

#findng cotour with 4 potentioncorner & ROI arount it
for contour in contours:
    primeter = cv2.arcLength(contour,True)
    approx = cv2.approxPolyDP(contour,0.01*primeter,True) #Approximate a polygonal curve
    print("Approx",len(approx))

    if len(approx)==4: 
        contour_with_license_plate = approx
        x,y,w,h = cv2.boundingRect(contour)
        license_plate = gray_img[y:y+h,x:x+w]

        cv2.imshow("Plate",license_plate)
        break

(thresh,license_plate) = cv2.threshold(license_plate,127,255,cv2.THRESH_BINARY)
#cv2.imshow("Plate",license_plate)

license_plate = cv2.bilateralFilter(license_plate,11,17,17) #Filter for Noise reduction
#Text Recognition
text = pytesseract.image_to_string(license_plate)

img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
img = cv2.putText(img,text,(x-100,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


cv2.imshow("plate",img)
print("License plate",text) #Final Licence plate in text