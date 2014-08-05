# -*- coding: utf-8 -*-
"""
Pruebas para el algoritmo propuesto kNN-Sparse
Tomás Larrain A.
6 de junio de 2014
"""

import numpy as np
import utils.ASRUtils as asr
import utils.miscUtils as miscUtils
import os
import time
import cv2

# Parámetros
m = 600				# Cantidad de patches seleccionados por foto para A
m2 = 150 			# Cantidad de patches para Matriz S
height = 100		# Alto del resize de la imagen
width = 100			# Ancho del resize de la imagen
a = 18				# Alto del patch
b = 18				# Ancho del patch
alpha = 0.5 		# Peso del centro
Q = 5				# Cluster Padres
R = 5 				# Cluser Hijos
sub = 1				# Subsample
sparseThreshold = 0 # Umbral para binarizar la representación sparse
cantPersonas = 20 	# Cantidad de personas para el experimento


# Inicializacion Variables
cantIteraciones = 100
porcAcumulado = 0
testTimeAcumulado = 0
trainTimeAcumulado = 0


# Datos de entrada del dataset
dataBase = "ORL"
rootPath = miscUtils.getDataBasePath(dataBase)
cantPhotos = miscUtils.photosPerPerson(rootPath)
cantPhotosDict = 2
cantPhotosSparse = cantPhotos-cantPhotosDict-1

U = asr.LUT(height,width,a,b) # Look Up Table
ii,jj = asr.grilla(height,width,a,b,m) # Generacion de esquinas superiores izquierdas aleatorias (i,j)

for it in range(cantIteraciones): # repite el experimento cantIteraciones veces
	
	print "Iteracion ", it+1, " de ", cantIteraciones
	print "Entrenando..."
	
	beginTime = time.time()

	# Entrenamiento: Diccionario A y Parches Sparse
	YC = np.array([])
	YP = np.array([])

	# Seleccion aleatoria de individuos
	idxPhoto, idxPerson = miscUtils.randomSelection(rootPath, cantPhotos, cantPersonas)
	
	##################################
	######### ENTRENAMIENTO ##########
	##################################

	######### CREACION DICCIONARIO ##########
	for i in range(cantPersonas):

		# Ruta de la persona i y lista de todas sus fotos
		route = os.path.join(rootPath, idxPerson[i])
		photos = os.listdir(route)
		
		Y = np.array([])
		
		for j in range(cantPhotosDict):
			
			routePhoto = os.path.join(route, photos[idxPhoto[j]]) # ruta de la foto j
			I = miscUtils.readScaleImage(routePhoto, width, height) # lectura de la imagen

			Yaux = asr.patches(I, ii, jj, U, a, b, alpha, sub) # extracción de parches
		
			# Concatenación de matrices Yaux
			Y = miscUtils.concatenate(Yaux, Y, 'vertical')

		YCaux,YPaux = asr.modelling(Y, Q, R) # Clusteriza la matriz Y en padres e hijos

		# Concatenación de matrices YC e YP
		YC = miscUtils.concatenate(YCaux, YC, 'vertical')
		YP = miscUtils.concatenate(YPaux, YP, 'vertical')

	# Inicializacion de variables
	Y = np.array([])
	Ysparse = np.array([])

	# Grilla de m2 cantidad de parches
	ii,jj = asr.grilla(height, width, a, b, m2)



	######### CREACION REPRESENTACIONES SPARSE ##########
	for i in range(cantPersonas):
		route = os.path.join(rootPath, idxPerson[i])
		photos = os.listdir(route)
		
		for j in range(cantPhotosSparse):
			idx = j+cantPhotosDict
			routePhoto = os.path.join(route, photos[idxPhoto[idx]])

			I = miscUtils.readScaleImage(routePhoto, width, height)
			# Generacion de esquinas superiores izquierdas aleatorias (i,j)
			Y = asr.patches(I, ii, jj, U, a, b, alpha, sub)
			
			alpha1 = asr.normL1_omp(Y, YC, R)
			Ysparse = miscUtils.concatenate(alpha1, Ysparse, 'horizontal')
			

	# Binarización representaciones sparse
	Ysparse = Ysparse.transpose()
	YsparseBinary = (Ysparse < -sparseThreshold) | (Ysparse > sparseThreshold) # por umbral
	# YsparseBinary = Ysparse != 0 # distintas de cero
	
	# Inicialización variables de control
	trainTime = time.time() - beginTime
	trainTimeAcumulado += trainTime
	aciertos = 0
	
	responses = miscUtils.responseVector(cantPersonas, idxPerson, cantPhotosSparse)
	

	##################################
	############ TESTING #############
	##################################
	
	print "Testing..."
	beginTime = time.time()
	
	for i in range(cantPersonas):
		# Ruta de la foto de testing
		route = os.path.join(rootPath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[cantPhotos-1]])
		
		I = miscUtils.readScaleImage(routePhoto, width, height) # lectura de la imagne
		Y = asr.patches(I, ii, jj, U, a, b, alpha, sub) # generación de parches

		
		alpha1 = asr.normL1_omp(Y, YC, R) # representación sparse
		
		# alpha1 = asr.normL1_lasso(Y, YC, R)
		
		# Inicialización variables de testing
		resto = float('inf')
		restoAux = float('inf')
		correcto = cantPersonas+1
		
		# Binarización representaciones sparse
		alpha1 = alpha1.transpose()
		alphaBinary = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral
		# alphaBinary = alpha1 != 0 # distintas de cero
		
		for j in range(cantPersonas*cantPhotosSparse):
			
			Yclass = YsparseBinary[j*m2:(j+1)*m2, :] # matriz sparse que representa la foto
			
			resta = np.abs(Yclass-alphaBinary) # valor absoluto de la resta
			restoAux = np.sum(resta) # cantidad de elementos no negativos de la resta

			# Encuentra la resta con menor error
			if restoAux < resto:
				correcto = responses[j]
				resto = restoAux

		# Compara con vector de clasificación ideal
		if int(correcto) == int(idxPerson[i]):
			aciertos += 1
	
	# Control de tiempo
	testTime = time.time() - beginTime
	testTimeAcumulado += testTime/cantPersonas	
	
	# Resultados
	print "Porcentaje Aciertos: " , float(aciertos)/cantPersonas*100, "%\n"	
	porcAcumulado += float(aciertos)/cantPersonas*100



# RESULTADOS FINALES
print "Experimento finalizado"
print "Tiempo de entrenamiento promedio: ", trainTimeAcumulado/cantIteraciones, " segundos"
print "Tiempo de testing promedio: ", testTimeAcumulado/cantIteraciones, " segundos/persona"
print "Porcentaje acumulado: ", porcAcumulado/cantIteraciones, "%"



