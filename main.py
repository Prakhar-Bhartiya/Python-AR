import cv2
import argparse
import imutils
import sys
import numpy as np

# ARUCO Basics : https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/aruco_basics.html

# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image containing ArUCo tag")
# ap.add_argument("-s", "--source", required=True,
# 	help="path to input source image that will be put on input")
# args = vars(ap.parse_args())


# load the input image from disk, resize it, and grab its spatial
# dimensions
print("[INFO] loading input image and source image...")
# image = cv2.imread(args["image"])
image = cv2.imread("Markers/marker2.png")
# image = imutils.resize(image, width=600)
(imgH, imgW) = image.shape[:2]
# load the source image from disk
# source = cv2.imread(args["source"])
source = cv2.imread("peter.jpg")


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



# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("[INFO] detecting markers...")
# arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_6X6_250"])
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
	parameters=arucoParams)
# if we have not found four markers in the input image then we cannot
# apply our augmented reality technique
# if len(corners) != 4:
# 	print("[INFO] could not find 4 corners...exiting")
# 	sys.exit(0)
print(len(corners))
# frame_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)
# cv2.imshow("Frame", frame_markers)
# cv2.waitKey(0)


# for cor, id in zip(corners, ids):
#     print("ID :", id)
#     print("corner ", cor)
#     print("-----")
# sys.exit(0)
# otherwise, we've found the four ArUco markers, so we can continue
# by flattening the ArUco IDs list and initializing our list of
# reference points
print("[INFO] constructing augmented reality visualization...")
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
imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
output = cv2.add(warpedMultiplied, imageMultiplied)
output = output.astype("uint8")

# show the input image, source image, output of our augmented reality
frame_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

marker_window = "Detected Marker"
cv2.namedWindow(marker_window)        # Create a named window
cv2.moveWindow(marker_window, 40,30)  # Move it to (40,30)
cv2.imshow(marker_window, frame_markers)


source_window = "Input Image"
cv2.namedWindow(source_window)        # Create a named window
cv2.moveWindow(source_window, 40,30)  # Move it to (40,30)
cv2.imshow(source_window, source)


output_window = "OpenCV AR Output"
cv2.namedWindow(output_window)        # Create a named window
cv2.moveWindow(output_window, 40,30)  # Move it to (40,30)
cv2.imshow(output_window, output)


cv2.waitKey(0)

"""python opencv_ar_image.py --image examples/input_02.jpg 
	--source sources/antelope_canyon.jpg """


if __name__ == "__main__":
    print("Running From Main")
