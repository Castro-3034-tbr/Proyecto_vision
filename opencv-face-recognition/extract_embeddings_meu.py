# EMPREGA:
# python extract_embeddings_meu.py --conf config/config.json


# importamos os paquete precisos
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import json
from json_minify import json_minify
import cv2
import os

# Analizador de argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path ao ficheiro de configuracion")
args = vars(ap.parse_args())

#Lemos o ficeheiro de configuracion
conf = json.loads(json_minify(open(args["conf"]).read()))

#Aqui solo empregamos un dos detectores. Podese elixir outro!
print("[INFO] cargando o detector de caras...")
modelFile = conf["model_caffe_path"]
configFile = conf["configfile_caffe_path"]
detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)


# cargamos o modelo para o codificador de caras
print("[INFO] cargando o modelo para codificacion ...")
embedder = cv2.dnn.readNetFromTorch(conf["embedding_model_path"])

# Lemos os cartafoles e imaxes na base de datos
print("[INFO] Analizando as caras caras da bd...")
imagePaths = list(paths.list_images(conf["dataset_path"]))

# Inicializamos as variables para almacenar
# os vectores de caracteristicas e nomes da persoas (nomes dos cartafoles na base de imaxes)
knownEmbeddings = []
knownNames = []

# inicializamos o contador de imaxes procesadas
total = 0

# lazo sobre todas as imaxes da base de datos 
for (i, imagePath) in enumerate(imagePaths):
	# extraemos o nome da persoa a partir do seu path
	print("[INFO] procesando imaxe {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	#Lemos a imaxe e a redimensionamos para ter unha anchura de 600 pixels (mantemos a relación de aspecto) e anotamos as novas dimensións da imaxe
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# A función blobFromImage é unha función que realiza o preprocesado da imaxe antes de aplicarlle a rede neuronal. Esta función realiza: resta a medida da imaxe, escala e, opcionalmente, aplica intercambio de canles de cor. 
	#Os valores medios no adestramento de ImageNet son asignados a R=103.93, G=116.77, e B=123.68.
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 117.0, 123.0), swapRB=False, crop=False)

	# aplicamos o detector de caras sobre a imaxe preprocesada
	detector.setInput(imageBlob)
	detections = detector.forward()

	# Comprobamos que detectamos, polo menos, unha imaxe
	if len(detections) > 0:
		# estamos asumindo que na imaxe so hai UNHA cara
		# por iso, enmarcaremos a cara con maior probabilidade 
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# aseguramonos que a detección con maior probabilidade tamén supera a probabilidade esixida para descartar deteccións débiles fixada polo usuario test 
		if confidence > conf["confidence"]:
			# achamos as coordenadas (x, y)do rectangulo que encadra a cara
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extraemos a rexión da cara (ROI) e anotamos as dimemsions da ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
		
			# aseguramonos que a altura e anchura da cara son suficientemente grandes
			if fW < 20 or fH < 20:
				continue
			
			# Preprocesamo a ROI da imaxe detectada para pasala a traves da rede que codifica as cara (obten un vector 128-d caracteristicas)
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# engadimos o nome da persona + mais a cara embebida (vector 128-d) a correspondte lista
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# grabamos a disco as caras embebidas + nomes 
print("[INFO] numero de codificacións {} ...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(conf["embeddings_path"], "wb")
f.write(pickle.dumps(data))
f.close()