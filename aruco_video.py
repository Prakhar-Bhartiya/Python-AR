from imutils.video import VideoStream
import imutils
import time 
import cv2
import sys
import numpy as np


ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_6X6_250"])
arucoParams = cv2.aruco.DetectorParameters_create()

source = cv2.imread("peter.jpg")

print("Starting Video Stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=1000)    
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    (imgW, imgH) = frame.shape[:2]

    if len(corners)>0 and all(ele in ids for ele in [2, 4, 12, 10]):
        ids = ids.flatten()
        refPts = []
        # loop over the IDs of the ArUco markers in top-left, top-right,
        # bottom-right, and bottom-left order
        for i in (2, 4, 12, 10):
            # grab the index of the corner with the current ID and append the
            # corner (x, y)-coordinates to our list of reference points
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(corners[j])
            refPts.append(corner)

        # unpack our ArUco reference points and use the reference points to
        # define the *destination* transform matrix, making sure the points
        # are specified in top-left, top-right, bottom-right, and bottom-left
        # order
        (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
        dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
        dstMat = np.array(dstMat)
        # grab the spatial dimensions of the source image and define the
        # transform matrix for the *source* image in top-left, top-right,
        # bottom-right, and bottom-left order
        (srcH, srcW) = source.shape[:2]
        srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
        # compute the homography matrix and then warp the source image to the
        # destination based on the homography
        (H, _) = cv2.findHomography(srcMat, dstMat)
        warped = cv2.warpPerspective(source, H, (imgW, imgH))

        # construct a mask for the source image now that the perspective warp
        # has taken place (we'll need this mask to copy the source image into
        # the destination)
        mask = np.zeros((imgH, imgW), dtype="uint8")
        cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
            cv2.LINE_AA)
        # this step is optional, but to give the source image a black border
        # surrounding it when applied to the source image, you can apply a
        # dilation operation
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, rect, iterations=2)
        # create a three channel version of the mask by stacking it depth-wise,
        # such that we can copy the warped source image into the input image
        maskScaled = mask.copy() / 255.0
        maskScaled = np.dstack([maskScaled] * 3)
        # copy the warped source image into the input image by (1) multiplying
        # the warped image and masked together, (2) multiplying the original
        # input image with the mask (giving more weight to the input where
        # there *ARE NOT* masked pixels), and (3) adding the resulting
        # multiplications together


        warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
        imageMultiplied = cv2.multiply(frame, 1.0 - maskScaled)
        output = cv2.add(warpedMultiplied, imageMultiplied)
        output = output.astype("uint8")

        frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        # cv2.imshow("Frame", frame_markers)
        frame = output
        print("Dectected Markers!!!")


    cv2.imshow("Input", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # cv2.waitKey(0)

cv2.destroyAllWindows()
vs.stop()



    