#!/usr/bin/env python3
#File Name: proyecto02.py
#Authors: David Peraza Gonzalez and Guillermo Rodriguez Obando
#Date: 28/07/2019
#Description: Proyecto2: Calibracion de camara 
#Course: Vision por computador - MSc - Cuatrimestre II - ITCR

#Required libraries
import numpy as np
import cv2
import json
import glob

from pynput import keyboard

#Global variables
state = 0		#begin


#Debugging purposes
#params = {}
#params['chessboard'] = []
#params['camera'] = []
#params['chessboard'].append({
#    'rows': 7,
#    'columns': 6,
#	'imgFolder': 'Images',
#	'imgBaseNm': 'left',
#	'imgBaseIdx': 12
#})

#with open('params.txt', 'w') as outfile:
#    json.dump(params, outfile)

#Read algorithms parameters from json file
with open('params.txt', 'r') as json_file:
	params = json.load(json_file)

for chess in params['chessboard']:
	rows = chess['rows']
	columns = chess['columns']
	imgFolder = chess['imgFolder']
	imgBaseNm = chess['imgBaseNm']
	imgBaseIdx = chess['imgBaseIdx']

#Keyboard listener
def on_press(key):
	global state
	try: k = key.char # single-char keys
	except: k = key.name # other keys
	if key == keyboard.Key.esc:
		state = 10	#exit
		return False # stop listener
	if k == '1': # capture mode
		state = 1
		return True
	if k == '2': # stored mode
		state = 2
		return True

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def CalFromCam():

	global rows
	global columns

	idxCam = 0

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((columns*rows,3), np.float32)
	objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	cap = cv2.VideoCapture(0)

	while(True):

		# Capture frame-by-frame
		ret, frame = cap.read()

		img = frame

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Display the resulting frame
		cv2.imshow('Proyecto2: Calibracion de camara capturando de camara. (Camara)', frame)

		key = cv2.waitKey(2)

		if key & 0xFF == ord('d'):
			break

		if key & 0xFF == ord('c'):
			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, (rows,columns), cv2.CALIB_CB_FILTER_QUADS)

			# If found, add object points, image points (after refining them)
			if ret == True:

				# Save original image for future used
				cv2.imwrite(imgFolder + "/chessCam" +  str(idxCam) + ".jpg", frame)

				#Increase camera index for files
				idxCam += 1

				objpoints.append(objp)

				corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners2)

				# Draw and display the corners
				img = cv2.drawChessboardCorners(img, (rows,columns), corners2, ret)				

				#Show chessboard with found points
				cv2.imshow('Proyecto2: Calibracion de camara capturando de camara. (Chessboard)', img)
				cv2.waitKey(500)
			else:
				print("Chessboard corners not found!\n")

	cv2.destroyAllWindows()

	camera_matrix = camera_matrix = cv2.initCameraMatrix2D(objpoints, imgpoints, gray.shape[::-1])  
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)

	np.savez('camera', mtx = mtx, dist = dist, rvecs = rvecs, tvecs = tvecs)
	
	params['camera'].append({
    	'K': mtx.tolist(),
		'dist': dist.tolist()
	})

	with open('params.txt', 'w') as outfile:
	    json.dump(params, outfile)
	

	img = cv2.imread(imgFolder + '/' + imgBaseNm + str(imgBaseIdx) + '.jpg')
	h,  w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# crop the image
	x, y, w, h = roi
	dst = dst[y : y + h, x : x + w]
	cv2.imwrite('calibresult.png', dst)

	# Load previously saved data
	with np.load('camera.npz') as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

	print("Press d in camera window when you are done projecting.\n")

	while(True):

		# Capture frame-by-frame
		ret, frame = cap.read()

		img = frame
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (rows, columns), cv2.CALIB_CB_FILTER_QUADS)

		if ret == True:

			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

			# Find the rotation and translation vectors.
			ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)

			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
			img = draw(img, corners2, imgpts)

		key = cv2.waitKey(20)

		if key & 0xFF == ord('d'):
			break

		cv2.imshow('Proyecto2: Calibracion de camara en modo almacenamiento. (Proyeccion)', img)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def CalFromDisc():

	global rows
	global columns
	global params

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((columns*rows,3), np.float32)
	objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob(imgFolder + '/*.jpg')

	for fname in images:
 
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (rows,columns), cv2.CALIB_CB_FILTER_QUADS)

		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners2)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (rows,columns), corners2, ret)
			cv2.imshow('Proyecto2: Calibracion de camara en modo almacenamiento. (Chessboard)',img)
			cv2.waitKey(500)

	cv2.destroyAllWindows()

	camera_matrix = camera_matrix = cv2.initCameraMatrix2D(objpoints, imgpoints, gray.shape[::-1])  
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)

	np.savez('camera', mtx = mtx, dist = dist, rvecs = rvecs, tvecs = tvecs)
	
	params['camera'].append({
    	'K': mtx.tolist(),
		'dist': dist.tolist()
	})

	with open('params.txt', 'w') as outfile:
	    json.dump(params, outfile)
	

	img = cv2.imread(imgFolder + '/' + imgBaseNm + str(imgBaseIdx) + '.jpg')
	h,  w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# crop the image
	x, y, w, h = roi
	dst = dst[y : y + h, x : x + w]
	cv2.imwrite('calibresult.png', dst)

	# Load previously saved data
	with np.load('camera.npz') as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

	print("Press s in camera window for saving image.\n")

	for fname in glob.glob(imgFolder + '/' + imgBaseNm + '*.jpg'):

		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (rows, columns), cv2.CALIB_CB_FILTER_QUADS)

		if ret == True:

			corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
			# Find the rotation and translation vectors.
			ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
			img = draw(img, corners2, imgpts)
			
			cv2.imshow('Proyecto2: Calibracion de camara en modo almacenamiento. (Proyeccion)', img)

			k = cv2.waitKey(0) & 0xFF

			if k == ord('s'):
				cv2.imwrite(fname[:6]+'.png', img)

	cv2.destroyAllWindows()

def main():

	global state

	lis = keyboard.Listener(on_press=on_press)
	lis.start() 				# start to listen on a separate thread
	
	while(state != 10):	#loop until exit

		if state == 0:
			print("This program performs camera calibration using different images of a chessboard taken with camera or stored in disc.\n")
			print("Select a calibration mode.\n")
			print("Press 1 to calibrate and project using camera.\n")
			print("Press 2 to calibrate and project using stored images (.jpg) in Images folder.\n")
			print("Press ESC for exit.\n")

			state = 5		#idle

		if state == 1:
			print("Capture mode active! Press c for selecting image to use in calibration.\n")
			print("Press d in camera window when you are done selecting.\n")
			CalFromCam()
			state = 5

		if state == 2:
			print("Stored mode active! Selecting all .jpg files in folder Images.\n")
			CalFromDisc()			
			state = 5

if __name__ == "__main__":
    main()
