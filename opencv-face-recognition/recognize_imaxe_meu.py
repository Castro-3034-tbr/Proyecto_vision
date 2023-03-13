#https://www.bogotobogo.com/python/pytut.php
#https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3.php

# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php

# EMPREGA
# python recognize_imaxe_meu.py --conf config/config.json --imaxe imaxes/xose.jpg



# importamos os paquetes precisos
from imutils.video import VideoStream
from imutils.video import FPS
import dlib
import numpy as np
import argparse
import imutils
import pickle
import json
from json_minify import json_minify
import time
import cv2
import os

# construimos os argumentos do programa
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path ao ficheiro de configuracion")
ap.add_argument("-i", "--imaxe", required=True,
	help="path e nome da imaxe")
args = vars(ap.parse_args())


#Lemos o ficeheiro de configuracion
conf = json.loads(json_minify(open(args["conf"]).read()))


#ETAPA1: MODELOS DE DETECTORES DE CARAS (4)
if conf["detect_model"]=="HAAR":
	#Detector de caras: Haar cascade de OPencv
	# haarcascade_frontalface_default.xml is in models directory.
	faceCascade = cv2.CascadeClassifier(conf["cascade_path"])
elif conf["detect_model"] == "HOG":
	#Detector de caras: Hog (histograma de orientacions) de dlib
	hogFaceDetector = dlib.get_frontal_face_detector()
elif conf["detect_model"] == "MMOD":
	# Detector de caras: MMOD de dlib
	dnnFaceDetector = dlib.cnn_face_detection_model_v1(conf["mmod_path"])
elif conf["detect_model"] == "DNN_CAFFE":
	# OpenCV DNN supporta 2 redes neuronais.
	# 1. FP16 a version orixinal implementada en caffe(5.4MB)
	# 2. Unha version cuantizada en 8 bit empregando Tensorflow (2.7MB)
    modelFile = conf["model_caffe_path"]
    configFile = conf["configfile_caffe_path"]
    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)
elif conf["detect_model"] == "DNN_8B":
    modelFile = conf["model_tesorflow_path"]
    configFile = conf["configfile_tensorflow_path"]
    detector = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
else:
	raise Exception("Tipo de modelo non soportado:" + conf["detect_model"])

# ETAPA 2: CODIFICACION DE CARAS
# cargamos o modelo para codificar as caras
print("[INFO] cargamos o codificador de caras ...")
embedder = cv2.dnn.readNetFromTorch(conf["embedding_model_path"])


# ETAPA 3: CLASIFICADOR DE CARAS
# clasificador e etiquetas (nomes das persoas)
recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())


##################FUNCIÓNS PARA APLICAR OS DETECTORES ##########
def detectFaceOpenCVDnn(detector, frame):
	# Construimos o blob dende a imaxe (prepocesado)
	imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0), swapRB=False, crop=False)

	# Localizamos as caras na imaxe
	detector.setInput(imageBlob)
	detections = detector.forward()

	# lazo sobre todas as deteccions
	for i in range(0, detections.shape[2]):
		# extraemos a confianza (i.e., probabilidade) asociado coa prediccion
		confidence = detections[0, 0, i, 2]

		# filtramos as deteccions debiles
		if confidence > conf["confidence"]:
			# acha as coordenadas (x, y) do rectangulo que acouta a cara
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extraemos a ROI da cara
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# aseguramos que ancho e alto da rexions e sufucientemente grande
			if fW < 20 or fH < 20:
				continue
            #Codificamos e recoñecemos a cara detectada
			name, proba = codifica_reconhece_caras(frame, face)
			
			# debuxamos a rexion rectangular coa probabiliade asociada
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			
	return frame



def detectFaceOpenCVHaar(faceCascade,frame, inHeight=300, inWidth=0):
	frameOpenCVHaar = frame.copy()
	frameHeight = frameOpenCVHaar.shape[0]
	frameWidth = frameOpenCVHaar.shape[1]
	if not inWidth:
		inWidth = int((frameWidth / frameHeight) * inHeight)
	scaleHeight = frameHeight / inHeight
	scaleWidth = frameWidth / inWidth
	frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
	frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(image=frameGray, minSize=(20, 20))
	for (x, y, w, h) in faces:
		x1, y1, x2, y2 = x, y, x + w, y + h
		cvRect = (int(x1 * scaleWidth), int(y1 * scaleHeight), int(x2 * scaleWidth), int(y2 * scaleHeight))
		# extraemos a ROI da cara
		face = frameOpenCVHaar[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]
		(fH, fW) = face.shape[:2]
		if fW < 20 or fH < 20:
			continue
		#Codificamos a cara detectada
		name, proba = codifica_reconhece_caras(frameOpenCVHaar, face)
		
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = cvRect[1] - 10 if cvRect[1] - 10 > 10 else cvRect[1] + 10
		cv2.rectangle(frame, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]),(0, 0, 255), 2)
		cv2.putText(frame, text, (cvRect[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	return frame

def detectFaceDlibHog(detector, frame, inHeight=300, inWidth=0):
	frameDlibHog = frame.copy()
	frameHeight = frameDlibHog.shape[0]
	frameWidth = frameDlibHog.shape[1]
	if not inWidth:
		inWidth = int((frameWidth / frameHeight)*inHeight)
	scaleHeight = frameHeight / inHeight
	scaleWidth = frameWidth / inWidth
	frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))
	frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
	faceRects = detector(frameDlibHogSmall, 0)
	print(frameWidth, frameHeight, inWidth, inHeight)
	for faceRect in faceRects:
		cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight), int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight) ]
		cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/150)), 4)
		
		# extraemos a ROI da cara
		face = frameDlibHog[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]
		(fH, fW) = face.shape[:2]
		if fW < 20 or fH < 20:
			continue
	
		#Codificamos a cara detectada
		name, proba = codifica_reconhece_caras(frameDlibHog, face)
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = cvRect[1] - 10 if cvRect[1] - 10 > 10 else cvRect[1] + 10
		cv2.rectangle(frame, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]),(0, 0, 255), 2)
		cv2.putText(frame, text, (cvRect[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	return frame


def detectFaceDlibMMOD(detector, frame, inHeight=300, inWidth=0):
	frameDlibMMOD = frame.copy()
	frameHeight = frameDlibMMOD.shape[0]
	frameWidth = frameDlibMMOD.shape[1]
	if not inWidth:
		inWidth = int((frameWidth / frameHeight)*inHeight)
	scaleHeight = frameHeight / inHeight
	scaleWidth = frameWidth / inWidth
	frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))
	frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
	faceRects = detector(frameDlibMMODSmall, 0)
	print(frameWidth, frameHeight, inWidth, inHeight)
	for faceRect in faceRects:
		cvRect = [int(faceRect.rect.left()*scaleWidth), int(faceRect.rect.top()*scaleHeight),int(faceRect.rect.right()*scaleWidth), int(faceRect.rect.bottom()*scaleHeight) ]
		cv2.rectangle(frameDlibMMOD, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/150)), 4)
		# extraemos a ROI da cara
		face = frameDlibMMOD[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]
		(fH, fW) = face.shape[:2]
		if fW < 20 or fH < 20:
			continue
		#Codificamos a cara detectada
		name, proba = codifica_reconhece_caras(frameDlibMMOD, face)
		text = "{}: {:.2f}%".format(name, proba * 100)
		y = cvRect[1] - 10 if cvRect[1] - 10 > 10 else cvRect[1] + 10
		cv2.rectangle(frame, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]),(0, 0, 255), 2)
		cv2.putText(frame, text, (cvRect[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	return frame

#######################################################
#Codificador e clasificador de caras
def codifica_reconhece_caras(frame, faceROI):
    # Preprocesamos a rexion para codificala
    faceBlob = cv2.dnn.blobFromImage(faceROI, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()
    
    # clasificamos as caras
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]
    
	#Devolvemos a etiqueta e a probabilidade da cara
    return name, proba


#Fixamos as propiedades da xanela onde se 
#visualizara a imaxe
windowName="Imaxe"
cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)

# redimensionamos o frame a 600 pixeles (mantemos relación de aspecto) e gardamos as novas dimensions
print("[INFO] lemos a imaxe dende disco {}...".format(args["imaxe"]))

image = cv2.imread(args["imaxe"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]

#Detectamos, codificamos e recoñecemos as caras detectadas
if conf["detect_model"]=="HAAR":
	image = detectFaceOpenCVHaar(faceCascade,image)
elif conf["detect_model"] == "HOG":
	image = detectFaceDlibHog(hogFaceDetector, image)
elif conf["detect_model"] == "MMOD":
	image = detectFaceDlibMMOD(dnnFaceDetector, image)
elif conf["detect_model"] == "DNN_CAFFE":
	image = detectFaceOpenCVDnn(detector, image)
elif conf["detect_model"] == "DNN_8B":
	image = detectFaceOpenCVDnn(detector, image)
else:
	raise Exception("Tipo de modelo non soportado:" + conf["detect_model"])

# visualizamos o framede saida
cv2.imshow(windowName, image)
cv2.waitKey(0)
cv2.destroyAllWindows()