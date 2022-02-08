import cv2

from random import randrange

# Load pretrained data of front face images from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose  an image to detect faces 
img = cv2.imread('family.jpg')

# Change to grey scale
greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


# Detect faces
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img) # detectMiltuScale detect objects of different sizes in the inout image

# Print coordinates of face
#print(face_coordinates) 

# Draw rectangle around the faces
(x, y, w, h) = face_coordinates[0]  #first index of 2d array. Used to detect one face

for (x, y, w, h) in face_coordinates:
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5) # 1st tuple - x, y coordinates. 2nd - width+x, height+y. 3rd - color code. Last number - thickness of rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5) #random color version

# Show image
cv2.imshow("Face detector", img)  #(window title, image)

    

# Keeps the image open until user presses a key
cv2.waitKey()

print("Code Completed")