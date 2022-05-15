import cv2
import numpy as np

vid = cv2.VideoCapture("eye_motion.mp4")

while 1:
    ret , frame = vid.read()
    if ret is False:
        break

    roi = frame[0:210 , 230:450]
    rows , cols , _ = roi.shape
    gray = cv2.cvtColor(roi , cv2.COLOR_BGR2GRAY)
    _ , threshold = cv2.threshold(gray , 3 , 255 , cv2.THRESH_BINARY_INV)#siyah yerleri beyaz beyaz yerleri siyah yapmak lazım ondan INV

    contours , _  = cv2.findContours(threshold , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours , key = lambda x: cv2.contourArea(x) , reverse=True)# contourlar counterArea ya göre karşılaştırılıp sıralanacak
                                                                                    #terse göre

    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi , (x,y) , (x+w,h+y) , (255,0,0) , 2)
        cv2.line(roi , (x + int(w/2), 0) , (x+ int(w/2), rows) , (0,255,0) , 2)
        cv2.line(roi , (0 , y+ int(h/2)) , (cols , y+ int(h/2)) , (0,255,0) , 2)
        break

    cv2.imshow("Roi" , roi)
    cv2.imshow("T_Roi" , threshold)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()