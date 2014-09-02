# -*- coding: utf-8 -*-
"""
Pruebas para el algoritmo propuesto kNN-Sparse
Tomás Larrain A.
6 de junio de 2014
"""

import numpy as np
import utils.ASRUtils as asr
import utils.miscUtils as miscUtils
import utils.displayUtils as displayUtils
import os
import time
import cv2
from sklearn.neighbors import KNeighborsClassifier



# Parámetros
m = 400					# Cantidad de patches seleccionados por foto para A
m2 = 400				# Cantidad de patches para Matriz S
height = 100			# Alto del resize de la imagen
width = 100				# Ancho del resize de la imagen
a = 18					# Alto del patch
b = 18					# Ancho del patch
alpha = 0.5 				# Peso del centro
Q = 10					# Cluster Padres
R = 5 					# Cluser Hijos
sub = 1					# Subsample
sparseThreshold = 0 	# Umbral para binarizar la representación sparse
distType = 'absDiff'	# Tipo de distancia a utilizar. Puede ser 'hamming' o 'euclidean'
useAlpha = True			# Usar alpha en el vector de cada patch


# Variables de display
display = False			# Desplegar resultados
dispWidth = 30			# Ancho de las imágenes desplegadas
dispHeight = 30 		# Alto de las imágenes desplegadas

# Variables kNN scikit
useScikit = False					# Usar o no el clasificador de Scikit
n_neighbors = np.array([1]) 		# Cantidad de vecinos a utilizar


# Inicializacion variables controlNombre 	Dirección	Costo/día	Total
porcAcumulado = np.zeros(len(n_neighbors))
testTimeAcumulado = 0
trainTimeAcumulado = 0


# Datos de entrada del dataset
dataBase = "ORL"
dataBasePath = miscUtils.getDataBasePath(dataBase)


# Datos de entrada del Test
cantIteraciones = 100 
cantPersonas = 20 		# Cantidad de personas para el experimento
# cantPhotosDict = 1
# cantPhotosSparse = 5
# cantPhotos = cantPhotosDict+cantPhotosSparse+1

cantPhotosDict = 4
cantPhotosSparse = 4 
cantPhotos = cantPhotosSparse+1
cantPhotosPerPerson = 10

idxPerson = miscUtils.personSelectionByPhotoAmount(dataBasePath, cantPhotos)

if len(idxPerson) < cantPersonas:
	print "no hay suficiente cantidad de personas para realizar el experimento"
	exit()

U = asr.LUT(height,width,a,b) # Look Up Table

iiDict, jjDict = asr.grilla_v2(height, width, a, b, m) # Grilla de m cantidad de parches
# iiDict, jjDict = asr.randomCorners(height, width, a, b, m) # esquinas aleatorias

iiSparse, jjSparse = asr.grilla_v2(height, width, a, b, m2) # Grilla de m2 cantidad de parches



for it in range(cantIteraciones): # repite el experimento cantIteraciones veces
	
	print "Iteracion ", it+1, " de ", cantIteraciones
	print "Entrenando..."
	
	beginTime = time.time()

	# Entrenamiento: Diccionario A y Parches Sparse
	YC = np.array([])
	YP = np.array([])

	# Seleccion aleatoria de individuos
	# idxPerson, idxPhoto = miscUtils.randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas)
	idxPerson, idxPhoto = miscUtils.randomSelectionOld(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas)
	
	# idxPerson = np.load("idxPersonMalo.npy")
	# idxPhoto = np.load("idxPhotoMalo.npy")

	##################################
	######### ENTRENAMIENTO ##########
	##################################
	
	######### CREACION DICCIONARIO ##########
	for i in range(cantPersonas):

		# Ruta de la persona i y lista de todas sus fotos
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		
		
		Y = np.array([])
		filaDict = np.array([])
		
		for j in range(cantPhotosDict):
			
			# routePhoto = os.path.join(route, photos[idxPhoto[i,j]]) # ruta de la foto j
			routePhoto = os.path.join(route, photos[idxPhoto[j]]) # ruta de la foto j
			I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagen
					
			Yaux = asr.patches(I, iiDict, jjDict, U, a, b, alpha, sub, useAlpha) # extracción de parches
		
			# Concatenación de matrices Yaux
			Y = miscUtils.concatenate(Yaux, Y, 'vertical')

		
		YCaux,YPaux = asr.modelling(Y, Q, R) # Clusteriza la matriz Y en padres e hijos
		
		# Concatenación de matrices YC e YP
		YC = miscUtils.concatenate(YCaux, YC, 'vertical')
		YP = miscUtils.concatenate(YPaux, YP, 'vertical')

	# Inicializacion de variables
	Y = np.array([])
	Ysparse = np.array([])

	
	######### CREACION REPRESENTACIONES SPARSE ##########
	
	for i in range(cantPersonas):
	
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		
		for j in range(cantPhotosSparse):
			# idx = j+cantPhotosDict
			idx = j
			# routePhoto = os.path.join(route, photos[idxPhoto[i,idx]])
			routePhoto = os.path.join(route, photos[idxPhoto[idx]])
			I = miscUtils.readScaleImageBW(routePhoto, width, height)
			
			alpha1 = asr.fingerprint(I, U, YC, iiSparse, jjSparse, R, a, b, alpha, sub, useAlpha)
			Ysparse = miscUtils.concatenate(alpha1, Ysparse, 'horizontal')
		

	Ysparse = Ysparse.transpose()
	
	if  distType != 'euclidean' and distType != 'chiSquare':  # Si la distancia elegida no es euclideana o chiSquare se binariza
		Ysparse = (Ysparse < -sparseThreshold) | (Ysparse > sparseThreshold) # por umbral
		# YsparseBinary = Ysparse != 0 # distintas de cero
	

	# Inicialización variables de control
	trainTime = time.time() - beginTime
	trainTimeAcumulado += trainTime
	
	responses = miscUtils.responseVector(cantPersonas, idxPerson, cantPhotosSparse)
	
	
	
	##################################
	############ TESTING #############
	##################################
	
	print "Testing..."
	beginTime = time.time()
	
	if useScikit:	
		aciertos, correctPhoto = asr.testing_Scikit(dataBasePath, idxPerson, idxPhoto, n_neighbors, width, height, 
			U, YC, Ysparse, iiSparse, jjSparse, R, a, b, alpha, sub, sparseThreshold, useAlpha, distType, responses)
	
	else:			
		# aciertos, correctPhoto = asr.testing(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Ysparse, iiSparse, jjSparse, R, a, b, 
		# 	alpha, sub, sparseThreshold, useAlpha, distType, responses)

		aciertos, correctPhoto = asr.testingOld(dataBasePath, cantPersonas, idxPerson, idxPhoto, width, height, U, YC, Ysparse, iiSparse, jjSparse, R, a, b, 
			alpha, sub, sparseThreshold, useAlpha, distType, responses)
		
	# Control de tiempo
	testTime = time.time() - beginTime
	testTimeAcumulado += testTime/cantPersonas	
	
	# Resultados
	
	if useScikit:
		print "Porcentaje Aciertos:"
		for n in range(len(n_neighbors)):
			print "k = " + str(n_neighbors[n]) + ": " + str(float(aciertos[n])/cantPersonas*100) + "%"
			porcAcumulado[n] += float(aciertos[n])/cantPersonas*100
			print "Porcentaje Acumlado k = ", n_neighbors[n], ":\t", str(float(porcAcumulado)/(it+1)) + "%\n"
	else:
		print "Porcentaje Aciertos: " , float(aciertos)/cantPersonas*100, "%"
		
		# if 	float(aciertos)/cantPersonas*100 < 75:
		# 	print "Guardando experimento..."
		# 	np.save("idxPersonMalo", idxPerson)
		# 	np.save("idxPhotoMalo", idxPhoto)
		# porcAcumulado += float(aciertos)/cantPersonas*100
		
		print "Porcentaje Acumlado: ", float(porcAcumulado)/(it+1), "%\n"	



	if display:
		results = displayUtils.generateResults(correctPhoto, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, 
			dataBasePath, dispWidth, dispHeight)
		
		allPhotos = displayUtils.generateAllPhotos(cantPersonas, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, 
			dataBasePath, dispWidth, dispHeight)
		displayUtils.displayResults(results, allPhotos)



	
# RESULTADOS FINALES
print "Experimento finalizado"
print "Base de Datos: ", dataBase
print "Tipo de distancia: ", distType
print "Se utilizó alpha: ", useAlpha
print "Cantidad de personas: ", cantPersonas
print "Fotos para diccionario: ", cantPhotosDict
print "Fotos para base de datos: ", cantPhotosSparse , "\n"

title = "Variables utilizadas:"
print title
print miscUtils.fixedLengthString(title, "m: " + str(m))
print miscUtils.fixedLengthString(title, "m2: " + str(m2))
print miscUtils.fixedLengthString(title, "height: " + str(height))
print miscUtils.fixedLengthString(title, "width: " +str(width))
print miscUtils.fixedLengthString(title, "a: " + str(a))
print miscUtils.fixedLengthString(title, "b: " + str(b))
print miscUtils.fixedLengthString(title, "alpha: " + str(alpha))
print miscUtils.fixedLengthString(title, "Q: " + str(Q))
print miscUtils.fixedLengthString(title, "R: " + str(R))
print miscUtils.fixedLengthString(title, "sub: " + str(sub))
print miscUtils.fixedLengthString(title, "sparseThreshold: " + str(sparseThreshold)) + "\n"

print "Tiempo de entrenamiento promedio: ", trainTimeAcumulado/cantIteraciones, " segundos/persona"
print "Tiempo de testing promedio: ", testTimeAcumulado/cantIteraciones, " segundos/persona"


if useScikit:
	print "Porcentaje acumulado:"
	for n in range(len(n_neighbors)):
		print "k = " + str(n_neighbors[n]) + ": " + str(porcAcumulado[n]/cantIteraciones) + "%\n"

else:
	print "Porcentaje acumulado: ", porcAcumulado[0]/cantIteraciones, "%\n"


print "Tiempo total del test: ", (testTimeAcumulado + trainTimeAcumulado)/60, " minutos"







