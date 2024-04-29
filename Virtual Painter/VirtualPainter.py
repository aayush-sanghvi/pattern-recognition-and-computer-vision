# Aayush Sanghvi & Yogi Shah
# PRCV Final project presentation 
# Date: 24th April 2024

#User Experience Guide: 
#Using one hand
# Drawing Hand: 
# --> Index: Howering
# --> Index + Middle: Drawing
# --> Index + Middle + Ring: Erasing for Onscreen whiteboard.
#Using Two hand(Drawing Hand + second hand): 
# Drawing hand: all same operation work independently.
# Second hand:
# --> Index + Middle: Settings Display => Drawing hand Index for selection.
#       - Brush Size Selection
#       - Eraser Size Selection
#       - Color Selection
# --> Index + Middle + Ring: Clear complete board.
# --> Index + Drawing Hand (all 4 fingers): capture the screenshot of the complete image. 




import cv2
import time
import HandTracker as htm
import numpy as np
import os

#Predefined values.
xp, yp = 0, 0
drawColor = (0, 0, 255)
eraseColor = (0, 0, 0)
brushSize = 10
eraseSize = 25

imgCanvas = np.zeros((720, 1280, 3), np.uint8)# defining canvas

cap=cv2.VideoCapture(0)
cap.set(3,1280)#width
cap.set(4,720)#height

detector = htm.handDetector(detectionCon=0.50,maxHands=2)#making object

while True:

    # 1. Import image
    success, img = cap.read()
    img=cv2.flip(img,1)#for neglecting mirror inversion
    click = 0
    # 2. Find Hand Landmarks
    img = detector.findHands(img)#using functions for connecting landmarks
    lmList = detector.findPosition(img)#using function to find specific landmark position,draw false means no circles on landmarks
    if len(lmList)!=0:
        
        #print(lmList)
        x1, y1 = lmList[8][1],lmList[8][2]# tip of index finger
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        
        # Index finger Up of Drawing hand it does nothing
        if fingers[1] and (not fingers[2]) and (not fingers[3]) and (not fingers[4]):
            xp,yp=x1,y1

        # Index + Middle finger Up of Drawing hand it will draw by using Index finger tip point.
        if fingers[1] and fingers[2] and (not fingers[3]) and (not fingers[4]):
            cv2.circle(img, (x1, y1), brushSize , drawColor, cv2.FILLED)#drawing mode is represented as circle
            #print("Drawing Mode")
            if xp == 0 and yp == 0:#initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                xp, yp = x1, y1 # so to avoid that we set xp=x1 and yp=y1
            #till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushSize)#gonna draw lines from previous coodinates to new positions 
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushSize)
            xp,yp=x1,y1
        # Index + Middle + Ring finger Up of Drawing hand it will erase by using Index finger tip point.
        elif fingers[1] and fingers[2] and fingers[3] and (not fingers[4]):
            cv2.circle(img, (x1, y1), eraseSize, eraseColor, cv2.FILLED)#drawing mode is represented as circle
            #print("Drawing Mode")
            if xp == 0 and yp == 0:#initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                xp, yp = x1, y1 # so to avoid that we set xp=x1 and yp=y1
            #till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also
            cv2.line(img, (xp, yp), (x1, y1), eraseColor, eraseSize)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), eraseColor, eraseSize)
            xp,yp=x1,y1
        #Checking if there are 2 hands
        if len(fingers)>5:
            
            # Index + Middle finger Up of second hand Opens the settings option listed below, which can be selected 
            # by hovering index finger on the option or the selection of your choice 
            # --> BrushSize selection
            # --> EraserSize selection
            # --> Color Selection
            if fingers[6] and fingers[7] and (not fingers[8]) and (not fingers[9]): 
                #Display of setting menu
                cv2.circle(img, (1150, 40), 20, (255,165,0), cv2.FILLED)
                cv2.circle(img, (1200, 40), 20, (0,255,0), cv2.FILLED)
                cv2.circle(img, (1250, 40), 20, (0,0,255), cv2.FILLED)
                cv2.putText(img, str("Color Options"), (1145, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                cv2.circle(img, (1030, 40), 5, (0,0,0), cv2.FILLED)
                cv2.circle(img, (1050, 40), 10, (0,0,0), cv2.FILLED)
                cv2.circle(img, (1080, 40), 15, (0,0,0), cv2.FILLED)
                cv2.putText(img, str("Brush Size"), (1020, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                cv2.circle(img, (880, 35), 10, (0,0,0), cv2.FILLED)
                cv2.circle(img, (920, 45), 20, (0,0,0), cv2.FILLED)
                cv2.circle(img, (980, 55), 30, (0,0,0), cv2.FILLED)
                cv2.putText(img, str("Eraser Size"), (880, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
                #Color selection
                if (x1 > 1130 and x1 < 1170) and (y1 > 30 and y1 < 50):
                    drawColor = (255,165,0) # Blue
                elif (x1 > 1180 and x1 < 1220) and (y1 > 30 and y1 < 50):
                    drawColor = (0,255,0) # Green 
                elif (x1 > 1230 and x1 < 1270) and (y1 > 30 and y1 < 50):
                    drawColor = (0,0,255) # Red
                #BrushSize Selection
                if (x1 > 1025 and x1 < 1035) and (y1 > 30 and y1 < 50):
                    brushSize = 5
                elif (x1 > 1040 and x1 < 1060) and (y1 > 30 and y1 < 50):
                    brushSize = 10 
                elif (x1 > 1070 and x1 < 1090) and (y1 > 30 and y1 < 50):
                    brushSize = 15
                #EraseSize Selection
                if (x1 > 875 and x1 < 885) and (y1 > 25 and y1 < 45):
                    eraseSize = 10 
                elif (x1 > 910 and x1 < 930) and (y1 > 30 and y1 < 55):
                    eraseSize = 20 
                elif (x1 > 965 and x1 < 995) and (y1 > 30 and y1 < 60):
                    eraseSize = 30
            
            # Clears the complete white board of all the drawings. 
            if fingers[6] and fingers[7] and fingers[8] and (not fingers[9]): 
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            # Flag the system for a Screenshot operation.
            if fingers[1] and fingers[2] and fingers[3] and fingers[4] and fingers[6] and (not fingers[7]) and (not fingers[8]) and (not fingers[9]):
                click = 1
                
               
    
    # 1 converting img to gray
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    # 2 converting into binary image and thn inverting
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)#on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask
    
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)#converting again to gray bcoz we have to add in a RGB image i.e img
    
    #add original img with imgInv ,by doing this we get our drawing only in black color
    img = cv2.bitwise_and(img,imgInv)
    
    #add img and imgcanvas,by doing this we get colors on img
    img = cv2.bitwise_or(img,imgCanvas)

    # if the Finger flag the Screen shot then it save that frame and give a 0.5 sec delay for the user to capture a photo.
    if click:
        cv2.imwrite(str(time.time_ns()/100000)+"_canvas.jpg",imgCanvas)
        cv2.imwrite(str(time.time()/100000)+".jpg",img)
        cv2.waitKey(500)
    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()