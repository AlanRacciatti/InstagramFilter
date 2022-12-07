import cv2
import numpy as np

# Read the input image and the two images you want to put in the cheeks
face_img = cv2.imread('face.jpg')
left_cheek = cv2.imread('penguin.png')
right_cheek = cv2.imread('penguin.png')

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect the face in the input image
faces = face_cascade.detectMultiScale(face_img, 1.1, 4)

# Iterate over the detected faces
for (x, y, w, h) in faces:
    # Resize the left and right cheek images
    resized_left_cheek = cv2.resize(left_cheek, (w//2, h//2))
    resized_right_cheek = cv2.resize(right_cheek, (w//2, h//2))

    # Get the region of interest (ROI) on the face image where the left cheek should be placed
    left_cheek_roi = face_img[y:y+resized_left_cheek.shape[0],
                              x:x+resized_left_cheek.shape[1]]

    # Get the region of interest (ROI) on the face image where the right cheek should be placed
    right_cheek_roi = face_img[y:y+resized_right_cheek.shape[0], x +
                               resized_right_cheek.shape[1]:x+resized_right_cheek.shape[1]*2]

    # Use NumPy's bitwise_and() method to merge the left cheek ROI and the resized left cheek
    face_img[y:y+resized_left_cheek.shape[0], x:x+resized_left_cheek.shape[1]
             ] = np.bitwise_and(left_cheek_roi, resized_left_cheek)

    # Use NumPy's bitwise_and() method to merge the right cheek ROI and the resized right cheek
    face_img[y:y+resized_right_cheek.shape[0], x+resized_right_cheek.shape[1]:x +
             resized_right_cheek.shape[1]*2] = np.bitwise_and(right_cheek_roi, resized_right_cheek)

# Display the output image
cv2.imwrite('output.jpg', face_img)
