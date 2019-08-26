# -*- coding: utf-8 -*-
"""
Set de pruebas para el algoritmo propuesto kNN-Sparse
Tomás Larrain A.
1 de octubre de 2014
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


LL = np.array([1, 2, 4, 6])
SCI = np.array([0, 0.1, 0.15, 0.2])

dataBases = ["FWM"]

for db, dataBase in enumerate(dataBases):

	dataBasePath, cantPhotosPerPerson = dataBaseUtils.getDataBasePath(dataBase)
	useAlpha = True			# Usar alpha en el vector de cada patch
	tanTriggs = False		# Utilizar normalizacion de Tan-Triggs
	
	if dataBase == 'Yale':
		tanTriggs = True

	cantPersonas = 20		# Cantidad de personas para el experimento
	cantPhotosDict = 4
	cantIteraciones = 100
	cantExperimentos = len(LL)*len(SCI)
	index = '_1_'
	testName = dataBase + '_' + "test" + index + str(cantPersonas) + '_' + str(cantPhotosDict) + ".txt"

	testPath = dataBaseUtils.getTestPath()

	testFile = open(os.path.join(testPath, testName), "w")


	header = miscUtils.testResultsHeader(dataBase, useAlpha, tanTriggs, cantPersonas, cantPhotosDict, cantIteraciones, cantExperimentos)
	print header
	testFile.write(header)

	trainline = 'Tiempo Train (seg)'
	testline = miscUtils.fixedLengthString(trainline, 'Tiempo Test (seg)')
	totalline = miscUtils.fixedLengthString(trainline, 'Tiempo Total (min)')
	porcline = miscUtils.fixedLengthString(trainline, 'Porcentaje')

	title = miscUtils.fixedLengthString(trainline, 'Experimento')
	mline = miscUtils.fixedLengthString(trainline, 'm')
	m2line = miscUtils.fixedLengthString(trainline, 'm2')
	heightline = miscUtils.fixedLengthString(trainline, 'height')
	widthline = miscUtils.fixedLengthString(trainline, 'width')
	aline = miscUtils.fixedLengthString(trainline, 'a')
	bline = miscUtils.fixedLengthString(trainline, 'b')
	alphaline = miscUtils.fixedLengthString(trainline, 'alpha')
	Qline = miscUtils.fixedLengthString(trainline, 'Q')
	Rline = miscUtils.fixedLengthString(trainline, 'R')
	Lline = miscUtils.fixedLengthString(trainline, 'L')
	subline = miscUtils.fixedLengthString(trainline, 'sub')
	STline = miscUtils.fixedLengthString(trainline, 'sparseThr')
	SCIline = miscUtils.fixedLengthString(trainline, 'SCIThr')
	personline = miscUtils.fixedLengthString(trainline, 'cantPersonas')
	photosDictline = miscUtils.fixedLengthString(trainline, 'cantPhotosDict')

	cont = 1


	print 'Base de datos ' + str(db+1) + ' de ' + str(len(dataBases))
	for L in LL:
		for SCIThreshold in SCI:

			# Parámetros
			m = 400					# Cantidad de patches seleccionados por foto para A
			m2 = 400				# Cantidad de patches para Matriz S
			s = 400					# Cantidad de parches para la votacion
			height = 100			# Alto del resize de la imagen
			width = 100				# Ancho del resize de la imagen
			a = 18					# Alto del patch
			b = 18					# Ancho del patch
			alpha = 0.5	 			# Peso del centro
			Q = 20					# Cluster Padres
			R = 10					# Cluser Hijos
			sub = 1					# Subsample
			sparseThreshold = 0 	# Umbral para binarizar la representación sparse
			
			title += '\t' + 'EXP' + str(cont)
			
			mline += '\t' + str(m)
			m2line += '\t' + str(m2)
			heightline += '\t' + str(height)
			widthline += '\t' + str(width)
			aline += '\t' + str(a)
			bline += '\t' + str(b)
			alphaline += '\t' + str(alpha)
			Qline += '\t' + str(Q)
			Rline += '\t' + str(R)
			Lline += '\t' + str(L)
			subline += '\t' + str(sub)
			STline += '\t' + str(sparseThreshold)
			SCIline += '\t' + str(SCIThreshold)
			personline += '\t' + str(cantPersonas)
			photosDictline += '\t' + str(cantPhotosDict)

			# Variables de display
			display = False			# Desplegar resultados
			dispWidth = 30			# Ancho de las imágenes desplegadas
			dispHeight = 30 		# Alto de las imágenes desplegadas


			# Inicializacion variables
			porcAcumulado = 0
			testTimeAcumulado = 0
			trainTimeAcumulado = 0

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
			print 'Experimento ' + str(cont) + '/' + str(cantExperimentos)
			cont += 1
			for it in range(cantIteraciones): # repite el experimento cantIteraciones veces
				
				print "Iteracion ", it+1, " de ", cantIteraciones
				
				
				# Seleccion aleatoria de individuos
				if cantPhotosPerPerson != 0:
					idxPerson, idxPhoto = dataBaseUtils.randomSelection_Old(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas)
				else:
					idxPerson, idxPhoto = dataBaseUtils.randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas)

				if dataBase == "ARx":
					idxPhoto = dataBaseUtils.randomSelectionPhoto_ARx(cantPhotos, cantPersonas)	


				##################################
				######### ENTRENAMIENTO ##########
				##################################

				beginTime = time.time()
				######### CREACION DICCIONARIO ##########
				YC = asr.generateDictionary(dataBasePath, idxPerson, idxPhoto, iiDict, jjDict, Q, R, U, width, height, a, b, alpha, sub, useAlpha, cantPhotosDict, tanTriggs)
				

				# Inicialización variables de control
				trainTime = time.time() - beginTime
				trainTimeAcumulado += trainTime
				
				
				##################################
				############ TESTING #############
				##################################
				
				
				beginTime = time.time()
				
				# aciertos = magisterUtils.testing_v2(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, m2, iiSparse, jjSparse, L, a, b, 
				# 	alpha, sub, sparseThreshold, useAlpha)

				aciertos = magisterUtils.testing_v3(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, m2, s, iiSparse, jjSparse, L, a, b, 
					alpha, sub, sparseThreshold, SCIThreshold, useAlpha, tanTriggs)
				

				# Control de tiempo
				testTime = time.time() - beginTime
				testTimeAcumulado += testTime/cantPersonas

				porcAcumulado += float(aciertos)/cantPersonas*100

			trainline += '\t' + "{:.2f}".format(trainTimeAcumulado/cantIteraciones)
			testline += '\t' + "{:.2f}".format(testTimeAcumulado/cantIteraciones)
			totalline += '\t' + "{:.2f}".format((testTimeAcumulado*cantPersonas + trainTimeAcumulado)/60)
			porcline += '\t' + "{:.2f}".format(porcAcumulado/cantIteraciones)
			
			results = miscUtils.testResults(cantPersonas, m, m2, height, width, a, b, alpha, Q, R, L, sub, sparseThreshold, SCIThreshold, trainTimeAcumulado, testTimeAcumulado, porcAcumulado, cantIteraciones)
			print results
			
	testFile.write(title + '\n') 
	testFile.write(mline + '\n') 
	testFile.write(m2line + '\n') 
	testFile.write(heightline + '\n')
	testFile.write(widthline + '\n')
	testFile.write(aline + '\n')
	testFile.write(bline + '\n')
	testFile.write(alphaline + '\n')
	testFile.write(Qline + '\n')
	testFile.write(Rline + '\n') 
	testFile.write(Lline + '\n') 
	testFile.write(subline + '\n') 
	testFile.write(STline + '\n')
	testFile.write(SCIline + '\n')
	testFile.write(personline + '\n')
	testFile.write(photosDictline + '\n')
	testFile.write(trainline + '\n')
	testFile.write(testline + '\n')
	testFile.write(totalline + '\n')
	testFile.write(porcline + '\n')
	testFile.close()

print "EXPERMIENTO FINALIZADO"



