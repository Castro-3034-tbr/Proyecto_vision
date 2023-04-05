import imutils
from imutils import paths
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import pickle
import json
import dlib
from json_minify import json_minify
import cv2
import os
import time


def capturar_imagen():
    """Funcion que se usa para capturar las imagenes de la persona que se le pasa por parametro"""
    global vs, frame

    # Creamos las variables locales
    total = 0

    # Pedimos el nombre de la persona
    nombre = input("Nombre de la persona: ")

    # Creamos la carpeta de la persona
    direccion = str("/home/castro/Escritorio/Proyecto_vision/mi_modelo/dataset/"+nombre)
    print("Carpeta creada: "+direccion)
    os.mkdir(direccion)
    time.sleep(2.0)

    # Realizamos las capturas de la persona
    for i in range(0, 20):
        # Leemos cada frame y lo redimensionamos para aplicar rapidamente el detector de caras
        frame = vs.read()
        orig = frame.copy()
        frame = imutils.resize(frame, width=400)
        
        # Salvamos el frame original a la carpeta de la persona
        p = os.path.sep.join([direccion, "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, orig)
        print("Captura guardada")
        total += 1
        time.sleep(0.5)
        
        


def extraer_embeddings():
    """Funcion que usamos para sacar el embedding de la cara que se le pasa por parametro"""

    global conf

    # DEfinimos las variables locales
    total_imagenes = 0

    # Eliximos el detector de caras
    print("[INFO] cargando o detector de caras...")
    modelFile = conf['model_caffe_path']
    configFile = conf['configfile_caffe_path']
    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # Cargamos el modelo para el codificador de caras
    print("[INFO] cargando el codificador de caras...")
    embedder = cv2.dnn.readNetFromTorch(conf["embedding_model_path"])

    # Leemos las carpetas de la base de datos
    print("[INFO] cargando las imagenes...")
    imagePaths = list(paths.list_images("dataset"))

    # Inicializamos las listas de las caras y los nombres
    knownEmbeddings = []
    knownNames = []

    # Analizamos cada una de las imagenes
    for (i, imagePath) in enumerate(imagePaths):
        # Extraemos el nombre de la persona de la ruta de la imagen
        print("[INFO] procesando imagen {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # Leemos la imagen y la redimensionamos para tener una estructura de 600 pixeles
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # Realizamos el preprocesado de la imagen
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Aplicamos el detector de caras para localizar las caras
        detector.setInput(imageBlob)
        detections = detector.forward()

        # Comprobamso que detectamos, polos menos, una cara
        if len(detections) > 0:
            # Extraemos la confianza maxima de la deteccion
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # Nos aseguramos que la confianza es mayor que el umbral
            if confidence > conf["confidence"]:
                # Buscamos las coordenados (x,y) del recuadro que rodea la cara
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Extraemos la cara de la imagen (ROI)
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Aseguramos que la cara es lo suficientemente grande
                if fW < 20 or fH < 20:
                    continue
                # Procesamos a ROI para extraer el embedding
                faceBlob = cv2.dnn.blobFromImage(
                    face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # Añadimos el nombre de la persona y su embedding a las listas
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total_imagenes += 1

    # Guardamos los embeddings en disco
    print("[INFO] serializando {} embeddings...".format(total_imagenes))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(conf["embeddings_path"], "wb")
    f.write(pickle.dumps(data))
    f.close()


def codifica_reconhece_caras(frame, faceROI):
    # Preprocesamos a rexion para codificala
    faceBlob = cv2.dnn.blobFromImage(
        faceROI, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
    embedder.setInput(faceBlob)
    vec = embedder.forward()

    # clasificamos as caras
    preds = recognizer.predict_proba(vec)[0]
    j = np.argmax(preds)
    proba = preds[j]
    name = le.classes_[j]
    #Devolvemos a etiqueta e a probabilidade da cara
    return name, proba


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

		#Añadimos la confianza al array
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
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	#Calculamos las media de las confianzas
	return frame, proba


def detectFaceOpenCVHaar(faceCascade, frame, inHeight=300, inWidth=0):
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
		cvRect = (int(x1 * scaleWidth), int(y1 * scaleHeight),int(x2 * scaleWidth), int(y2 * scaleHeight))
		# extraemos a ROI da cara
		face = frameOpenCVHaar[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]
		(fH, fW) = face.shape[:2]
		if fW < 20 or fH < 20:
			continue
		#Codificamos a cara detectada
		name, proba = codifica_reconhece_caras(frameOpenCVHaar, face)

		text = "{}: {:.2f}%".format(name, proba * 100)
		y = cvRect[1] - 10 if cvRect[1] - 10 > 10 else cvRect[1] + 10
		cv2.rectangle(frame, (cvRect[0], cvRect[1]),(cvRect[2], cvRect[3]), (0, 0, 255), 2)
		cv2.putText(frame, text, (cvRect[0], y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	return frame, proba


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
        cvRect = [int(faceRect.left()*scaleWidth), int(faceRect.top()*scaleHeight),int(faceRect.right()*scaleWidth), int(faceRect.bottom()*scaleHeight)]
        cv2.rectangle(frameDlibHog, (cvRect[0], cvRect[1]), (cvRect[2],cvRect[3]), (0, 255, 0), int(round(frameHeight/150)), 4)

		# extraemos a ROI da cara
        face = frameDlibHog[cvRect[1]:cvRect[3], cvRect[0]:cvRect[2]]
        (fH, fW) = face.shape[:2]
        if fW < 20 or fH < 20:
            continue

		#Codificamos a cara detectada
        name, proba = codifica_reconhece_caras(frameDlibHog, face)
        text = "{}: {:.2f}%".format(name, proba * 100)
        y = cvRect[1] - 10 if cvRect[1] - 10 > 10 else cvRect[1] + 10
        cv2.rectangle(frame, (cvRect[0], cvRect[1]),(cvRect[2], cvRect[3]), (0, 0, 255), 2) 
        cv2.putText(frame, text, (cvRect[0], y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    return frame , proba




# Obtenemos los valores de la entrada por consola
ap = argparse.ArgumentParser()
ap.add_argument("-ca", "--cascade", required=True,help="path al detector de caras Haar cascade")
ap.add_argument("-co", "--conf", required=True,help="path al fichero de configuracion")
args = vars(ap.parse_args())

# Cargamos en OpenCV el Haar cascade desde disco
detector = cv2.CascadeClassifier(args["cascade"])

# Leemos el fichero de configuracion
conf = json.loads(json_minify(open(args["conf"]).read()))

#Preconfiguramos el detector de caras
if conf["detect_model"] == "HAAR":
	# Detector de caras: Haar cascade de OPencv
	# haarcascade_frontalface_default.xml is in models directory.
	faceCascade = cv2.CascadeClassifier(conf["cascade_path"])
elif conf["detect_model"] == "HOG":
	# Detector de caras: Hog (histograma de orientacions) de dlib
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

#Cargamos el modelo para codificar las caras
print("[INFO] cargamos o codificador de caras ...")
embedder = cv2.dnn.readNetFromTorch(conf["embedding_model_path"])

#Cargamos el clasificador e etiquetas de las caras
recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())

# Inicializamos la camara
print("[INFO] Inicializamos la camara...")
vs = VideoStream(src=0).start()
time.sleep(5.0)

# Creamos la pantalla
windowName = "Fame"
cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)

# Incializamos los FPS
fps = FPS().start()

# Creamos el bucle de deteccion
while True:
    # Leemos cada frame
    frame = vs.read()
    # Redimensionamos el frame para que sea mas rapido
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Eligimos el detector de caras y detectamos las caras codificamos as caras
    if conf["detect_model"] == "HAAR":
        frame, proba = detectFaceOpenCVHaar(faceCascade, frame)
    elif conf["detect_model"] == "HOG":
        frame, proba = detectFaceDlibHog(hogFaceDetector, frame)
    #elif conf["detect_model"] == "MMOD":
        #frame, proba = detectFaceDlibMMOD(dnnFaceDetector, frame)
    elif conf["detect_model"] == "DNN_CAFFE":
        frame, proba = detectFaceOpenCVDnn(detector, frame)
    elif conf["detect_model"] == "DNN_8B":
        frame, proba = detectFaceOpenCVDnn(detector, frame)
    else:
        raise Exception("Tipo de modelo non soportado:" + conf["detect_model"])
    
    # Actualizamos los FPS
    fps.update()
    
    # Mostramos los FPS de salida 
    cv2.imshow(windowName, frame)
    key = cv2.waitKey(1) & 0xFF
    
    print("Probabilidad: " + str(proba*100))
    
    # Si la probilidad de 0.5, vamos a guardar la nueva cara 
    if proba < 0.5:
        #Capturamos las imagenes 
        capturar_imagen()
        #Obtenemos los embeddings 
        extraer_embeddings()
    
    # Si pulsamos la tecla 'q' salimos del bucle
    if key == ord("q"):
        break

# paramos os contadores de tempo e frames FPS
fps.stop()
print("[INFO] tempo transcurrido: {:.2f}".format(fps.elapsed()))
print("[INFO] aprox. FPS: {:.2f}".format(fps.fps()))


# Hacemos de limpieza 
print("[INFO] Limpiamos el dataset...")
cv2.destroyAllWindows()
vs.stop()
