# -*- coding: utf-8 -*-
"""
Pruebas con LFWa
Tomás Larrain A.
17 de octubre de 2014
"""


import numpy as np
import utils.ASRUtils as asr
import utils.miscUtils as miscUtils
import utils.displayUtils as displayUtils
import utils.magisterUtils as magisterUtils
import utils.dataBaseUtils as dataBaseUtils
import utils.imageUtils as imageUtils
import utils.testUtils as testUtils
import os
import time
import cv2
from scipy import io


# Parámetros
m = 1225 				# Cantidad de patches seleccionados por foto para A
m2 = 789 				# Cantidad de patches para Matriz S
s = m2
height = 100			# Alto del resize de la imagen
width = 100				# Ancho del resize de la imagen
w = 20					# Alto del patch
alpha = 0.5	 			# Peso del centro
Q = 20					# Cluster Padres
R = 20					# Cluser Hijos
L = 2					# Cantidad de elementos en repr. sparse
sub = 2
SCIThreshold = 0.1		# Umbral de seleccion de patches
sparseThreshold = 0
tanTriggs = False		# Utilizar normalizacion de Tan-Triggs
useAlpha = True

dataBase = "LFWa"
dataBasePath, cantPhotosPerPerson = dataBaseUtils.getDataBasePath(dataBase)
idxPerson = dataBaseUtils.getPersonIDs(dataBasePath)




cantPhotosDict = 10

U = asr.LUT(height, width, w) # Look Up Table

iiDict, jjDict = asr.grilla_v2(height, width, w, m) # Grilla de m cantidad de parches
iiSparse, jjSparse = asr.grilla_v2(height, width, w, m2) # Grilla de m2 cantidad de parches

idxPhotoDict = np.array(range(cantPhotosDict))
idxPhotoDict = np.tile(idxPhotoDict, (len(idxPerson), 1))

allPhotos = dataBaseUtils.totalPhotos(dataBasePath, idxPerson[1])
testPhotos = allPhotos[10:]


# YC = asr.generateDictionary(dataBasePath, idxPerson, idxPhotoDict, iiDict, jjDict, Q, R, U, width, height, w, alpha, sub, useAlpha, cantPhotosDict, tanTriggs)
# np.save('LFWaDictionary.npy', YC)

YC = np.load('LFWaDictionary.npy')
aciertos = testUtils.testingLFWa(dataBasePath, idxPerson, width, height, U, YC, Q, R, s, iiSparse, jjSparse, L, w, alpha, sub, 
	sparseThreshold, SCIThreshold, useAlpha, tanTriggs)


# Control de tiempo
testTime = time.time() - beginTime
testTimeAcumulado += testTime/cantPersonas	


# Resultados
porcAcumulado += float(aciertos.sum())/cantPersonas*100
# porcAcumuladoS += float(aciertosSin_S.sum())/cantPersonas*100

print "Porcentaje Aciertos: " , float(aciertos.sum())/cantPersonas*100, "%"	
print "Porcentaje Acumlado: ", float(porcAcumulado)/(it+1), "%"	




