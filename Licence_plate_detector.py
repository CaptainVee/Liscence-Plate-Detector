import cv2
import numpy as np
import matplotlib.pyplot as plt


def display(img):
    fig = plt.figure(figsize = (12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)


car = cv2.imread('Desktop/042AI/OpenCV Course (Udemy)/Computer-Vision-with-Python/DATA/car_plate.jpg')
car = cv2.cvtColor(car, cv2.COLOR_BGR2RGB)
display(car)

### Drawing a rectangle on the lisence plate
lisence = cv2.CascadeClassifier('Desktop/042AI/OpenCV Course (Udemy)/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_russian_plate_number.xml')

def detect_Lplate(img):
    
    plate_img = img.copy()
    
    plate_rectangle = lisence.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 5)
    
    for (x,y,w,h) in plate_rectangle:
        cv2.rectangle(plate_img, (x,y), (x+w, y+h), (255,0,0), 2)
            
    return plate_img

result = detect_Lplate(car)
plt.imshow(result)      


#### BLURRING THE LICENCE PLATE
def detect_blur_plate(img):
    
    plate_img = img.copy()
    roi = img.copy()
    
    plate_rectangle = lisence.detectMultiScale(plate_img, scaleFactor = 1.2, minNeighbors = 5)
    
    for (x,y,w,h) in plate_rectangle:
        roi = roi[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 9)
        plate_img[y:y+h, x:x+w] = blurred_roi
       
    return plate_img

result = detect_blur_plate(car)
plt.imshow(result)      


