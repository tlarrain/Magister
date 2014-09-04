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


# Parámetros
m = 400					# Cantidad de patches seleccionados por foto para A
m2 = 400				# Cantidad de patches para Matriz S
height = 100			# Alto del resize de la imagen
width = 100				# Ancho del resize de la imagen
a = 18					# Alto del patch
b = 18					# Ancho del patch
alpha = 0.5	 			# Peso del centro
Q = 25					# Cluster Padres
R = 20					# Cluser Hijos
L = 1 					# Cantidad de elementos en repr. sparse
sub = 1					# Subsample
sparseThreshold = 0 	# Umbral para binarizar la representación sparse
useAlpha = True			# Usar alpha en el vector de cada patch

# Variables de display
display = False			# Desplegar resultados
dispWidth = 30			# Ancho de las imágenes desplegadas
dispHeight = 30 		# Alto de las imágenes desplegadas


# Inicializacion variables controlNombre 	Dirección	Costo/día	Total
porcAcumulado = 0
testTimeAcumulado = 0
trainTimeAcumulado = 0


# Datos de entrada del dataset
dataBase = "AR"
dataBasePath, cantPhotosPerPerson = miscUtils.getDataBasePath(dataBase)

# Datos de entrada del Test
cantIteraciones = 100
cantPersonas = 20 		# Cantidad de personas para el experimento


cantPhotosDict = 4
cantPhotos = cantPhotosDict+1

idxTestPhoto = cantPhotosDict 
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
	
	

	# Seleccion aleatoria de individuos
	# idxPerson, idxPhoto = miscUtils.randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas)	
	idxPerson, idxPhoto = miscUtils.randomSelectionOld(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas)
	

	##################################
	######### ENTRENAMIENTO ##########
	##################################

	beginTime = time.time()
	######### CREACION DICCIONARIO ##########
	YC = asr.generateDictionary(dataBasePath, idxPerson, idxPhoto, iiDict, jjDict, Q, R, U, width, height, a, b, alpha, sub, useAlpha, cantPhotosDict)
	
	# Inicialización variables de control
	trainTime = time.time() - beginTime
	trainTimeAcumulado += trainTime
	
	
	##################################
	############ TESTING #############
	##################################
	
	print "Testing..."
	beginTime = time.time()
	
	aciertos = asr.testing_v2(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Ysparse, iiSparse, jjSparse, L, a, b, 
		alpha, sub, sparseThreshold, useAlpha)
	
	# Control de tiempo
	testTime = time.time() - beginTime
	testTimeAcumulado += testTime/cantPersonas	
	
	# Resultados	
	print "Porcentaje Aciertos: " , float(aciertos)/cantPersonas*100, "%"	
	porcAcumulado += float(aciertos)/cantPersonas*100
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
print "Se utilizó alpha: ", useAlpha
print "Cantidad de personas: ", cantPersonas
print "Fotos para diccionario: ", cantPhotosDict, "\n"

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