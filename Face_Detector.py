import cv2

from random import randrange

# Load pretrained data of front face images from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

''' 
#IMAGES

# Choose  an image to detect faces 
img = cv2.imread('Johnson.jpg')


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

# Display image with faces
cv2.imshow("Face detector", img)  #(window title, image)

    

# Keeps the image open until user presses a key
cv2.waitKey() #empty parameter- waits inifintely till a key is pressed or it automatically hits a key

print("Code Completed")
'''

#VIDEO

# Capture video from webcam
webcam = cv2.VideoCapture(0) # 0 is to access default webcamera

# Iterate forever through frames because it is in real time
while (True):
    # Read current frame
    sucessful_frame_read , frame = webcam.read() #frame is what we need

    # Change to grey scale
    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    #Display video
    cv2.imshow("Face detector", greyscaled_img)  #(window title, image)

    cv2.waitKey(1) #Delays for 1 millisecond before hitting a key itself 

    print("Code Completed")
