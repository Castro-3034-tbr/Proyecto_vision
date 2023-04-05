# #https://www.bogotobogo.com/python/pytut.php
#https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3.php

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#https://dataaspirant.com/2017/05/22/random-forest-algorithm-machine-learing/
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# 
# EMPREGA
# python train_model_meu.py --conf config/config.json


# importamos os paquetes necesarios
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle
import json
from json_minify import json_minify



#Analizamos os argumentos de entrada
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path ao ficheiro de configuracion")
args = vars(ap.parse_args())

#Lemos o ficeheiro de configuracion
conf = json.loads(json_minify(open(args["conf"]).read()))

# Cargamos as caras embebidas
print("[INFO] cargando as caras embebidas...")
data = pickle.loads(open(conf["embeddings_path"], "rb").read())

# codificamos as etiquetas de texto entre 0 e n_clases-1
print("[INFO] codificanco as etiquetas...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Entremos o modelos para aceptar caracteristicas de dimesnion 128-d das caras embebidas e producimos o recoñecemento de caras
print("[INFO] modelo seleccionado...")
recognizer = None
if conf["clasificador_model"] == "KNN":
    print ("clasificador kNN")
    recognizer = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
elif conf["clasificador_model"] == "SVN":
    print ("Clasificador SVN")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
elif conf["clasificador_model"] == "RF":
    print ("Clasificador Random Forest")
    recognizer = RandomForestClassifier()
else:
    raise Exception("Tipo de modelo non soportado:" + conf["clasificador_model"])

print("[INFO] entrenando o modelo...")
recognizer.fit(data["embeddings"], labels)

# Escribimos a disco o modelo do recoñecedor de caras entrenado
f = open(conf["recognizer_path"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Escribimos as etiquetas codificadas a disco
f = open(conf["le_path"], "wb")
f.write(pickle.dumps(le))
f.close()
