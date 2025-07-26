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
contour,new = cv2.findContours(canny_edge.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contour = sorted(contour,key=cv2.contourArea,reverse=True)[:30]

#Initialize the license plate contour
contour_with_licence_plate = None
license = None
x=None
y=None
w=None
h=None

#contour blanck contours on
contour_img = img.copy()
#Draw contours on the blanck
cv2.drawContours(contour_img,contour,-1,(0,0,255),2)
cv2.imshow("img with countors & Lince plate",contour_img)
