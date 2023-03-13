# 
#https://becominghuman.ai/face-detection-using-opencv-with-haar-cascade-classifiers-941dbb25177
# https://medium.com/dataseries/face-recognition-with-opencv-haar-cascade-a289b6ff042a

# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

# https://medium.com/dataseries/face-recognition-with-opencv-haar-cascade-a289b6ff042a
# EMPREGA
# python construir_base_caras.py --cascade haarcascade_frontalface_default.xml --output dataset/xose

# importamos os paquetes precisos
import imutils # conda install pip; pip install imutils
from imutils.video import VideoStream
import argparse
import time
import cv2
import os

# Definimos os argumentos do programa
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "path a onde reside o haar cascade")
ap.add_argument("-o", "--output", required=True,
	help="path ao directorio de saida")
args = vars(ap.parse_args())

# Cargamos en OpenCV o Haar cascade dende disco
detector = cv2.CascadeClassifier(args["cascade"])

# Inicializamos o fluxo de video
print("[INFO] Inicializamos a cámara...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
total = 0

# bucle sobre os frames de video
while True:
	# lemos cada frame e redimensionamolo para aplicar rapidamente o detector de caras
	frame = vs.read()
	orig = frame.copy()
	frame = imutils.resize(frame, width=400)

	# Aplicamos o detector de caras no frame convertido a imaxe de gris
	rects = detector.detectMultiScale(
		cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30))

	# bucle sobre todas as caras detectadas na imaxe e debuxamos un rectángulo verde sobre elas
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Visualizamos o frame modificado
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# Se presionas `k` salvamos o frame orixinal a disco para a nosa base de datos 
	if key == ord("k"):
		p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
		cv2.imwrite(p, orig)
		total += 1
		print("Captura gardada")

	#Se presionamos `q` saimos do buble
	elif key == ord("q"):
		break

# facemos un pouco de limpeza
print("[INFO] {} imaxes de caras gardadas".format(total))
print("[INFO] limpando...")
cv2.destroyAllWindows()
vs.stop()