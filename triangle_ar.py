import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = 'chessboard.avi'
K = np.array([[662.38304095,   0.,         617.30697704] ,
 [  0.,         672.73161008, 404.49037234] ,
 [  0.,           0.,           1.        ]])
dist_coeff = np.array( [ 0.14900088,  0.14887876,  0.04133815, -0.00206308, -0.34159169])
board_pattern = (8, 5)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Prepare 2D triangle points in 3D space (on the Z=0 plane)
triangle_points = board_cellsize * np.array([[3, 2, 0], [5, 2, 0], [4, 3, 0]])

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation 
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the triangle on the image
        triangle_2d, _ = cv.projectPoints(triangle_points, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(triangle_2d)], True, (0, 255, 255), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

# Cleanup
video.release()
cv.destroyAllWindows()
