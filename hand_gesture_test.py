import mediapipe as mp
"""
An example of using ArUco markers with OpenCV.
"""

import cv2
import sys
from cv2 import aruco
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

source = cv2.imread('peter.jpg')
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        scale_percent = 100 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # org_frame = frame
        # Check if frame is not empty
        if not ret:
            continue

        # Auto rotate camera
        #frame = cv2.autorotate(frame, device)

        # Convert from BGR to RGB
        hand_frame = frame.copy()
        results = hands.process(cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
            #     print(
            #     f'Index finger tip coordinates: (',
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width}, '
            #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height})'
            # )
            # val_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            # if val_x > 720:
            #   playsound("./sounds/C.mp3")
                
                mp_drawing.draw_landmarks(
                    hand_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                frame = hand_frame
        

        
        
                corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
                if len(corners) > 0 and all(ele in ids for ele in [2, 4, 12, 10]):
                    # for c in corners :
                    #     x1 = (c[0][0][0], c[0][0][1]) 
                    #     x2 = (c[0][1][0], c[0][1][1]) 
                    #     x3 = (c[0][2][0], c[0][2][1]) 
                    #     x4 = (c[0][3][0], c[0][3][1])   
                    #     im_dst = frame
                    #     size = source.shape
                    #     pts_dst = np.array([x1, x2, x3, x4])
                    #     pts_src = np.array(
                    #                    [
                    #                     [0,0],
                    #                     [size[1] - 1, 0],
                    #                     [size[1] - 1, size[0] -1],
                    #                     [0, size[0] - 1 ]
                    #                     ],dtype=float
                    #                    )
                    ids = ids.flatten()
                    refPts = []

                    for i in (2, 4, 12, 10):
                        # grab the index of the corner with the current ID and append the
                        # corner (x, y)-coordinates to our list of reference points
                        j = np.squeeze(np.where(ids == i))
                        corner = np.squeeze(corners[j])
                        refPts.append(corner)

                    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
                    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
                    dstMat = np.array(dstMat)
            
                    (srcH, srcW) = source.shape[:2]
                    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

                    pts_src = srcMat
                    pts_dst = dstMat
                        
                    h, status = cv2.findHomography(pts_src, pts_dst)
                    temp = cv2.warpPerspective(source.copy(), h, (frame.shape[1], frame.shape[0])) 
                    cv2.fillConvexPoly(frame, pts_dst.astype(int), 0, 16)
                    org_frame = cv2.add(frame, temp)
                    cv2.imshow('frame', org_frame)
        else:
            cv2.imshow('frame', frame)
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release everything:
cap.release()
cv2.destroyAllWindows()
