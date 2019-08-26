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
import utils.magisterUtils as magisterUtils
import utils.dataBaseUtils as dataBaseUtils
import os
import time
import cv2


# Parámetros
m = 400					# Cantidad de patches seleccionados por foto para A
m2 = 400				# Cantidad de patches para Matriz S
s = 400 				# Cantidad de parches para la votacion
height = 100			# Alto del resize de la imagen
width = 100				# Ancho del resize de la imagen
a = 18					# Alto del patch
b = 18					# Ancho del patch
alpha = 0.5	 			# Peso del centro
Q = 20					# Cluster Padres
R = 10					# Cluser Hijos
L = 6					# Cantidad de elementos en repr. sparse
sub = 1					# Subsample
sparseThreshold = 0 	# Umbral para binarizar la representación sparse
SCIThreshold = 0.05		# Umbral de seleccion de patches
useAlpha = True			# Usar alpha en el vector de cada patch
tanTriggs = False		# Utilizar normalizacion de Tan-Triggs



# Variables de display
display = False			# Desplegar resultados
dispWidth = 30			# Ancho de las imágenes desplegadas
dispHeight = 30 		# Alto de las imágenes desplegadas


# Inicializacion variables
porcAcumulado = 0
testTimeAcumulado = 0
trainTimeAcumulado = 0


# Datos de entrada del dataset
dataBase = "AR"

dataBasePath, cantPhotosPerPerson = dataBaseUtils.getDataBasePath(dataBase)

# Datos de entrada del Test
cantIteraciones = 1
cantPersonas = 20		# Cantidad de personas para el experimento


cantPhotosDict = 4
cantPhotos = cantPhotosDict+1

idxTestPhoto = cantPhotosDict 
idxPerson = dataBaseUtils.personSelectionByPhotoAmount(dataBasePath, cantPhotos)

if len(idxPerson) < cantPersonas:
	print "no hay suficiente cantidad de personas para realizar el experimento"
	exit()

U = asr.LUT(height,width,a,b) # Look Up Table

iiDict, jjDict = asr.grilla_v2(height, width, a, b, m) # Grilla de m cantidad de parches
# iiDict, jjDict = asr.randomCorners(height, width, a, b, m) # esquinas aleatorias

iiSparse, jjSparse = asr.grilla_v2(height, width, a, b, m2) # Grilla de m2 cantidad de parches
# iiSparse, jjSparse = asr.randomCorners(height, width, a, b, m) # esquinas aleatorias

for it in range(cantIteraciones): # repite el experimento cantIteraciones veces
	
	print "Iteracion ", it+1, " de ", cantIteraciones
	print "Entrenando..."
	
	# Seleccion aleatoria de individuos
	if cantPhotosPerPerson != 0:
		idxPerson, idxPhoto = dataBaseUtils.randomSelection_Old(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas)
	else:
		idxPerson, idxPhoto = dataBaseUtils.randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas)


	if dataBase == "ARx":
		idxPhoto = dataBaseUtils.randomSelectionPhoto_ARx(cantPhotos, cantPersonas)	

	idxPerson = np.load('idxPerson.npy')
	idxPhoto = np.load('idxPhoto.npy')
	
	##################################
	######### ENTRENAMIENTO ##########
	##################################

	beginTime = time.time()

	######### CREACION DICCIONARIO ##########
	YC = asr.generateDictionary(dataBasePath, idxPerson, idxPhoto, iiDict, jjDict, Q, R, U, width, height, a, b, alpha, sub, useAlpha, cantPhotosDict, tanTriggs)
	
	trainTime = time.time() - beginTime
	trainTimeAcumulado += trainTime
	
	
	##################################
	############ TESTING #############
	##################################
	
	print "Testing..."
	beginTime = time.time()
	
	aciertos = magisterUtils.testing_v3(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, m2, s, iiSparse, jjSparse, L, a, b, 
		alpha, sub, sparseThreshold, SCIThreshold, useAlpha, tanTriggs)
	

	# Control de tiempo
	testTime = time.time() - beginTime
	testTimeAcumulado += testTime/cantPersonas	
	

	# Resultados	
	print "Porcentaje Aciertos: " , float(aciertos)/cantPersonas*100, "%"	
	porcAcumulado += float(aciertos)/cantPersonas*100
	print "Porcentaje Acumlado: ", float(porcAcumulado)/(it+1), "%\n"	

	#np.save('idxPerson.npy', idxPerson)
	#np.save('idxPhoto.npy', idxPhoto)
	if display:
		# results = displayUtils.generateResults(correctPhoto, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, 
		# 	dataBasePath, dispWidth, dispHeight)
		
		allPhotos = displayUtils.generateAllPhotos(cantPersonas, cantPhotosDict, idxPhoto, idxPerson, 
			dataBasePath, dispWidth, dispHeight)
		
		cv2.namedWindow("experimento")
		cv2.imshow("experimento", allPhotos)
		cv2.waitKey()
		# displayUtils.displayResults(results, allPhotos)



header = miscUtils.testResultsHeader(dataBase, useAlpha, tanTriggs, cantPersonas, cantPhotosDict, cantIteraciones)	
results = miscUtils.testResults(cantPersonas, m, m2, height, width, a, b, alpha, Q, R, L, sub, sparseThreshold, SCIThreshold, trainTimeAcumulado, testTimeAcumulado, porcAcumulado, cantIteraciones)
print header
print results
