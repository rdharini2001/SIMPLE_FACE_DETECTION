import cv2
import sys
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)
img_counter = 0
while True:
 ret , frame = video.read()
 gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 k = cv2.waitKey(1)
 faces = face_cascade.detectMultiScale(gray , scaleFactor = 1.5, minNeighbors = 5, minSize = (30,30),flags = cv2.CASCADE_SCALE_IMAGE)
 for (x,y,w,h) in faces:
      cv2.rectangle(frame,(x,y),(x+w , y+h),(0,255,0),2)
      cv2.imshow("Face_Detection",frame)
 if k%256 == 27:
      break
 elif k%256 == 32:
      img_name = "face{}.png".format(img_counter)
      cv2.imwrite(img_name,frame)
      img_counter += 1
video.release()
cv2.destroyAllWindows()
