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
alpha = 0.5 			# Peso del centro
Q = 5					# Cluster Padres
R = 5 					# Cluser Hijos
sub = 1					# Subsample
sparseThreshold = 0 	# Umbral para binarizar la representación sparse
distType = 'hamming'	# Tipo de distancia a utilizar. Puede ser 'hamming', 'euclidean' o 'chiSquare'
n_neighbors = 1 		# Cantidad de vecinos a utilizar
useAlpha = True			# Usar alpha en el vector de cada patch
display = False			# Desplegar resultados
dispWidth = 30			# Ancho de las imágenes desplegadas
dispHeight = 30 		# Alto de las imágenes desplegadas

# Inicializacion variables control
cantIteraciones = 1
porcAcumulado = 0
testTimeAcumulado = 0
trainTimeAcumulado = 0
# neigh = KNeighborsClassifier(n_neighbors=n_neighbors,metric=distType)


# Datos de entrada del dataset
dataBase = "AR"
dataBasePath = miscUtils.getDataBasePath(dataBase)



cantPersonas = 20 		# Cantidad de personas para el experimento
cantPhotosDict = 5
cantPhotosSparse = 20
cantPhotos = cantPhotosDict+cantPhotosSparse+1

idxTestPhoto = cantPhotos-1

idxPerson = miscUtils.personSelectionByPhotoAmount(dataBasePath, cantPhotos)

U = asr.LUT(height,width,a,b) # Look Up Table

iiDict,jjDict = asr.grilla_v2(height, width, a, b, m) # Grilla de m cantidad de parches
iiSparse,jjSparse = asr.grilla_v2(height, width, a, b, m2) # Grilla de m2 cantidad de parches



for it in range(cantIteraciones): # repite el experimento cantIteraciones veces
	
	print "Iteracion ", it+1, " de ", cantIteraciones
	print "Entrenando..."
	
	beginTime = time.time()

	# Entrenamiento: Diccionario A y Parches Sparse
	YC = np.array([])
	YP = np.array([])

	# Seleccion aleatoria de individuos
	idxPerson, idxPhoto = miscUtils.randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas)

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
			
			routePhoto = os.path.join(route, photos[idxPhoto[i,j]]) # ruta de la foto j
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
		
		filaSparse = np.array([])
		for j in range(cantPhotosSparse):
			idx = j+cantPhotosDict
			
			routePhoto = os.path.join(route, photos[idxPhoto[i,idx]])
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
	aciertos = 0
	
	responses = miscUtils.responseVector(cantPersonas, idxPerson, cantPhotosSparse)
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior

	# neigh.fit(Ysparse, responses) # entrenador del algoritmo
	
	##################################
	############ TESTING #############
	##################################
	
	print "Testing..."
	beginTime = time.time()
			
	for i in range(cantPersonas):
		# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = asr.fingerprint(I, U, YC, iiSparse, jjSparse, R, a, b, alpha, sub, useAlpha)
		
		# Inicialización variables de testing
		resto = float('inf')
		corrPhoto = cantPersonas+1
		
		# Binarización representaciones sparse
		alpha1 = alpha1.transpose()
		if  distType != 'euclidean' and distType != 'chiSquare':
			alpha1 = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral
		# alphaBinary = alpha1 != 0 # distintas de cero
		
		np.save("alpha1",alpha1)
		
		corrPhoto, corrID = asr.clasifier(Ysparse, alpha1, responses, distType)
		correctPhoto[i,0] = corrPhoto

		
		# Compara con vector de clasificación ideal
		if int(corrID) == int(idxPerson[i]):
			aciertos += 1
			correctPhoto[i,1] = 1
			
	
	# Control de tiempo
	testTime = time.time() - beginTime
	testTimeAcumulado += testTime/cantPersonas	
	
	# Resultados
	print "Porcentaje Aciertos: " , float(aciertos)/cantPersonas*100, "%\n"	
	porcAcumulado += float(aciertos)/cantPersonas*100

	if display:
		results = displayUtils.generateResults(correctPhoto, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, dataBasePath, dispWidth, dispHeight)
		allPhotos = displayUtils.generateAllPhotos(cantPersonas, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, dataBasePath, dispWidth, dispHeight)
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
print "Porcentaje acumulado: ", porcAcumulado/cantIteraciones, "%\n"

print "Tiempo total del test: ", (testTimeAcumulado + trainTimeAcumulado)/60, " minutos"







