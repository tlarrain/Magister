# -*- coding: utf-8 -*-
"""
Método principal del algoritmo
Tomás Larrain A.
9 de octubre de 2014
"""

import numpy as np
import ASRUtils as asr
import miscUtils
import displayUtils
import magisterUtils
import dataBaseUtils
import imageUtils
import os
import time
import cv2
from sklearn.neighbors import NearestNeighbors


def mainAlgorithm(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold, tanTriggs, dataBase, cantIteraciones,
 cantPersonas, cantPhotosDict, display, displayResults, dispHeight, dispWidth, sensibilidad, grilla):

	# Variables fijas
	sparseThreshold = 0
	s = m2
	sub = 2
	useAlpha = True
	distThreshold = 0.3

	# Inicializacion variables
	porcAcumulado = 0
	porcAcumuladoAlt = 0
	testTimeAcumulado = 0
	trainTimeAcumulado = 0
	tiemposAcumulado = 0
	itResults = np.zeros(cantIteraciones)

	dataBasePath, cantPhotosPerPerson = dataBaseUtils.getDataBasePath(dataBase)
	cantPhotos = cantPhotosDict+1
	# print dataBaseUtils.photosPerPerson(dataBasePath)
	idxTestPhoto = cantPhotosDict 
	idxPerson = dataBaseUtils.personSelectionByPhotoAmount(dataBasePath, cantPhotos)

	if len(idxPerson) < cantPersonas:
		print "no hay suficiente cantidad de personas para realizar el experimento"
		exit()

	U = asr.LUT(height, width, w) # Look Up Table

	if grilla == True:
		iiDict, jjDict = asr.grilla_v2(height, width, w, m) # Grilla de m cantidad de parches
		iiSparse, jjSparse = asr.grilla_v2(height, width, w, m2) # Grilla de m2 cantidad de parches
	
	else:
		iiDict, jjDict = asr.randomCorners(height, width, w, m) # esquinas aleatorias
		iiSparse, jjSparse = asr.randomCorners(height, width, w, m2) # esquinas aleatorias


	# Experimentos sensibilidad
	if sensibilidad == True:
		idxPersonFull = np.load('idxPerson.npy')
		idxPhotoFull = np.load('idxPhoto.npy')

	for it in range(cantIteraciones): # repite el experimento cantIteraciones veces

		variables = miscUtils.testVariables(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold)
		
		if display:		
			print variables
		
		print "Iteracion ", it+1, " de ", cantIteraciones
		print "Entrenando..."
		
		# Seleccion aleatoria de individuos
		if cantPhotosPerPerson != 0:
			idxPerson, idxPhoto = dataBaseUtils.randomSelection_Old(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas)
		else:
			idxPerson, idxPhoto = dataBaseUtils.randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas)


		if dataBase == "ARx":
			idxPhoto = dataBaseUtils.randomSelectionPhoto_ARx(cantPhotos, cantPersonas)	

		# Experimentos sensibilidad
		if sensibilidad == True:
			idxPerson = idxPersonFull[it,:]
			idxPhoto = idxPhotoFull[it*cantPersonas:(it+1)*cantPersonas,:]
	

		
		##################################
		######### ENTRENAMIENTO ##########
		##################################
		
		beginTime = time.time()

		######### CREACION DICCIONARIO ##########
		# print idxPhoto
		YC = asr.generateDictionary(dataBasePath, idxPerson, idxPhoto, iiDict, jjDict, Q, R, U, width, height, w, alpha, sub, useAlpha, cantPhotosDict, tanTriggs)
		
		neigh = NearestNeighbors(1, metric='euclidean')
		neigh.fit(YC)
		trainTime = time.time() - beginTime
		trainTimeAcumulado += trainTime
		
		
		##################################
		############ TESTING #############
		##################################
		
		print "Testing..."
		beginTime = time.time()
		
		# inicio = time.time()
		# aciertos = testing(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, s, iiSparse, jjSparse, L, w, 
		# 	alpha, sub, neigh, sparseThreshold, SCIThreshold, distThreshold, useAlpha, tanTriggs)
		
		# print "tiempo de demora anterior: ", time.time()-inicio, " segundos"

	# 	aciertos, aciertosAlt = testingDoble(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, s, iiSparse, jjSparse, L, w, alpha, sub, neigh, 
	# 		sparseThreshold, SCIThreshold, distThreshold, useAlpha, tanTriggs)
		
		inicio = time.time()		
		aciertosAlt, tiempos = testingAdaptive(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, s, iiSparse, jjSparse, L, w, 
			alpha, sub, neigh, sparseThreshold, SCIThreshold, distThreshold, useAlpha, tanTriggs)

		# print "tiempo de demora nuevo: ", time.time()-inicio, " segundos"

		accuracy = float(aciertosAlt.sum())/cantPersonas*100
		# Control de tiempo
		testTime = time.time() - beginTime
		testTimeAcumulado += testTime/cantPersonas	
		
		tiemposAcumulado += tiempos/cantPersonas
		# Resultados
		# porcAcumulado += float(aciertos.sum())/cantPersonas*100
		porcAcumuladoAlt += accuracy
		# if float(aciertos.sum())/cantPersonas*100 < 90:
		# 	np.save('idxPersonMalo.npy', idxPerson)
		# 	np.save('idxPhotoMalo.npy', idxPhoto)
		
		itResults[it] = accuracy

		if displayResults:	
			# print "Porcentaje Aciertos: " , float(aciertos.sum())/cantPersonas*100, "%"	
			print "Porcentaje Aciertos: " , accuracy, "%\n"
			# print "Porcentaje Acumlado: ", float(porcAcumulado)/(it+1), "%"	
			print "Porcentaje Acumlado: ", float(porcAcumuladoAlt)/(it+1), "%\n"
		#np.save('idxPerson.npy', idxPerson)
		#np.save('idxPhoto.npy', idxPhoto)
		
		if display:
		
			allPhotos = displayUtils.generateAllPhotos(cantPersonas, cantPhotosDict, idxPhoto, idxPerson, 
				dataBasePath, dispWidth, dispHeight, aciertos)
			
			cv2.namedWindow("experimento")
			cv2.imshow("experimento", allPhotos)
			cv2.waitKey()
		
	
	return trainTimeAcumulado, testTimeAcumulado, tiemposAcumulado, itResults

def mainLongTest(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold, tanTriggs, dataBase, 
	cantIteraciones, cantPersonas, cantPhotosDict, display, displayResults, dispHeight, dispWidth, sensibilidad, grilla, fecha):
	
	if sensibilidad == True:
		m = 1225 				# Cantidad de patches seleccionados por foto para A
		m2 = 900				# Cantidad de patches para Matriz S
		height = 100			# Alto del resize de la imagen
		width = 100				# Ancho del resize de la imagen
		w = 20					# Alto del patch
		alpha = 0.5   			# Peso del centro
		Q = 50					# Cluster Padres
		R = 40					# Cluser Hijos
		L = 4					# Cantidad de elementos en repr. sparse
		SCIThreshold = 0.1		# Umbral de seleccion de patches
		tanTriggs = False		# Utilizar normalizacion de Tan-Triggs
		grilla = True
		dataBase = "AR"

	variables = np.array([1, 3, 5])
	
	dataBasePath, cantPhotosPerPerson = dataBaseUtils.getDataBasePath(dataBase)
	
	if dataBase == 'Yale':
		tanTriggs = True


	
	cantExperimentos = len(variables)
	
	testPath = dataBaseUtils.getTestPath()
	dbIdxFile = os.path.join(testPath, 'cont_'+str(dataBase)+'.npy')
	
	if not os.path.isfile(dbIdxFile):
		index = 1
		np.save(dbIdxFile, index)

	else:
		index = np.load(dbIdxFile)
		
	indexNext = index + 1
	np.save(dbIdxFile, indexNext)

	testName = dataBase + '_' + "test" + '_' + str(index) + '_' + str(cantPersonas) + '_' + str(cantPhotosDict) + ".txt"
	testFile = open(os.path.join(testPath, testName), "w")

	header = miscUtils.testResultsHeader(dataBase, tanTriggs, grilla, cantPersonas, cantPhotosDict, cantIteraciones, fecha, cantExperimentos=cantExperimentos)
	testFile.write(header)

	trainline = 'Tiempo Train (seg)'
	
	
	
	title = miscUtils.fixedLengthString(trainline, 'Experimento')
	mline = miscUtils.fixedLengthString(trainline, 'm')
	m2line = miscUtils.fixedLengthString(trainline, 'm2')
	wline = miscUtils.fixedLengthString(trainline, 'w')
	alphaline = miscUtils.fixedLengthString(trainline, 'alpha')
	Qline = miscUtils.fixedLengthString(trainline, 'Q')
	Rline = miscUtils.fixedLengthString(trainline, 'R')
	Lline = miscUtils.fixedLengthString(trainline, 'L')
	SCIline = miscUtils.fixedLengthString(trainline, 'SCIThr')
	porcline = miscUtils.fixedLengthString(trainline, 'Porcentaje')
	stdline = miscUtils.fixedLengthString(trainline, 'std')
	patchesSelline = miscUtils.fixedLengthString(trainline, 'patches Sel')

	# Lineas con tiempos de cada partes
	lectline = miscUtils.fixedLengthString(trainline, "Lectura: ") 
	extPatchline = miscUtils.fixedLengthString(trainline, "Ext. patches: ") 
	adaptIdxline = miscUtils.fixedLengthString(trainline, "Adapt. Idx: ")
	fpline = miscUtils.fixedLengthString(trainline, "Fingerprint: ") 
	maxIdxline = miscUtils.fixedLengthString(trainline, "Max Idx FP: ") 
	fpSelline = miscUtils.fixedLengthString(trainline, "Seleccion FP: ") 
	classline = miscUtils.fixedLengthString(trainline, "Clasificacion: ") 
	testline = miscUtils.fixedLengthString(trainline, 'Tiempo Test (seg)')

	totalline = miscUtils.fixedLengthString(trainline, 'Tiempo Total (min)')

	cont = 1


	for cantPhotosDict in variables:
		
		header = miscUtils.testResultsHeader(dataBase, tanTriggs, grilla, cantPersonas, cantPhotosDict, cantIteraciones, fecha, cantExperimentos=cantExperimentos)
		print header

		print 'Experimento ', cont, ' de ', cantExperimentos	
		
		variables = miscUtils.testVariables(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold)
		print variables
		
		title += '\t' + 'EXP' + str(cont)	
		mline += '\t' + str(m)
		m2line += '\t' + str(m2)
		wline += '\t' + str(w)
		alphaline += '\t' + str(alpha)
		Qline += '\t' + str(Q)
		Rline += '\t' + str(R)
		Lline += '\t' + str(L)
		SCIline += '\t' + str(SCIThreshold)
		

		# Variables de display
		display = False			# Desplegar resultados
		dispWidth = 30			# Ancho de las imágenes desplegadas
		dispHeight = 30 		# Alto de las imágenes desplegadas

		inicio = time.time()
		
		trainTimeAcumulado, testTimeAcumulado, tiemposAcumulado, itResults = mainAlgorithm(m, m2, height, width, w, alpha, Q, R, L, 
			SCIThreshold, tanTriggs, dataBase, cantIteraciones, cantPersonas, cantPhotosDict, display, displayResults, dispHeight, dispWidth, sensibilidad, grilla)

		totalTime = time.time() - inicio

		cantIteraciones = itResults.shape[0]
		porcAcumulado = np.mean(itResults)
		

		porcline += '\t' + "{:.2f}".format(porcAcumulado)
		stdline += '\t' + "{:.2f}".format(np.std(itResults))
		patchesSelline += '\t' + "{:.2f}".format(tiemposAcumulado[7]/cantIteraciones)

		trainline += '\t' + "{:.3f}".format(trainTimeAcumulado/cantIteraciones)
		lectline += '\t' + "{:.3f}".format(tiemposAcumulado[0]/cantIteraciones) 
		extPatchline += '\t' + "{:.3f}".format(tiemposAcumulado[1]/cantIteraciones)  
		adaptIdxline += '\t' + "{:.3f}".format(tiemposAcumulado[2]/cantIteraciones) 
		fpline += '\t' + "{:.3f}".format(tiemposAcumulado[3]/cantIteraciones)   
		maxIdxline += '\t' + "{:.3f}".format(tiemposAcumulado[4]/cantIteraciones)  
		fpSelline += '\t' + "{:.3f}".format(tiemposAcumulado[5]/cantIteraciones)   
		classline += '\t' + "{:.3f}".format(tiemposAcumulado[6]/cantIteraciones)   
		testline += '\t' + "{:.3f}".format(testTimeAcumulado/cantIteraciones)
		
		totalline += '\t' + "{:.2f}".format(totalTime/60)
		
		
		timeResult = miscUtils.timeResults(tiemposAcumulado, cantIteraciones)
		results = miscUtils.testResults(cantPersonas, trainTimeAcumulado, testTimeAcumulado, itResults)
		
		print results
		print timeResult

		cont += 1 

		
	testContent = ''		
	testContent += (title + '\n') 
	testContent += (mline + '\n') 
	testContent += (m2line + '\n') 
	testContent += (wline + '\n')
	testContent += (alphaline + '\n')
	testContent += (Qline + '\n')
	testContent += (Rline + '\n') 
	testContent += (Lline + '\n') 
	testContent += (SCIline + '\n')
	testContent += (porcline + '\n')
	testContent += (stdline + '\n')
	testContent += (patchesSelline + '\n\n')
	testContent += (trainline + '\n\n')
	testContent += (lectline + '\n') 
	testContent += (extPatchline + '\n')  
	testContent += (adaptIdxline + '\n') 
	testContent += (fpline + '\n')   
	testContent += (maxIdxline + '\n')  
	testContent += (fpSelline + '\n')   
	testContent += (classline + '\n')   
	testContent += (testline + '\n\n')
	testContent += (totalline + '\n')
	
	testFile.write(testContent)
	testFile.close()

	
	
	print "EXPERMIENTO FINALIZADO"
	return testContent



def testing(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, s, ii, jj, L, w, alpha, sub, neigh, 
	sparseThreshold, SCIThreshold, distThreshold, useAlpha, tanTriggs):
	# testing kNNSparse_v1.1 OFICIAL
	cantPersonas = len(idxPerson)
	idxTestPhoto = idxPhoto.shape[1]-1
	aciertos = np.zeros(cantPersonas)

	for i in range(cantPersonas):
		# Ruta de la foto de testing
		inicio = time.time()
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		I = imageUtils.readScaleImage(routePhoto, width, height, tanTriggs=tanTriggs) # lectura de la imagen
		
		inicio = time.time()
		Y = asr.patches(I, ii, jj, U, w, alpha, sub, useAlpha)
	
		
		alpha1 = magisterUtils.fingerprint(Y, YC, L)

		alpha1 = alpha1.transpose()
		
		inicio = time.time()
		ganador = clasifier(alpha1, Q, R, s, L, SCIThreshold, sparseThreshold, cantPersonas)
		
		if i == ganador:
			aciertos[i] = 1

	
	return aciertos


def testingAdaptive(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, s, ii, jj, L, w, alpha, sub, neigh, 
	sparseThreshold, SCIThreshold, distThreshold, useAlpha, tanTriggs):
	# testing kNNSparse_v1.1 OFICIAL
	cantPersonas = len(idxPerson)
	idxTestPhoto = idxPhoto.shape[1]-1
	aciertos = np.zeros(cantPersonas)
	
	photoTime = 0
	patchesTime = 0
	adaptiveTime = 0
	fpTime = 0
	maxIdxTimeC = 0 
	fpSelectTimeC = 0 
	idTimeC = 0
	cantSelectC = 0
	for i in range(cantPersonas):
		# Ruta de la foto de testing
		inicio = time.time()
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		I = imageUtils.readScaleImage(routePhoto, width, height, tanTriggs=tanTriggs) # lectura de la imagne
		photoTime += time.time()-inicio


		inicio = time.time()
		Y = asr.patches(I, ii, jj, U, w, alpha, sub, useAlpha)
		patchesTime += time.time()-inicio


		inicio = time.time()
		LimMat = asr.adaptiveDictionary_v4(Y, YC, Q, R)
		adaptiveTime += time.time()-inicio

		inicio = time.time()
		alpha1 = magisterUtils.fingerprintAdaptive(Y, YC, LimMat, L, R, cantPersonas)
		# alpha1 = magisterUtils.fingerprintAdaptive(Y, YC, LimMat, L, R, cantPersonas, neigh, distThreshold, True)
		fpTime += time.time()-inicio
		
		ganador, maxIdxTime, fpSelectTime, idTime, cantSelect = clasifier(alpha1, 1, R, s, L, SCIThreshold, sparseThreshold, cantPersonas)
		
		cantSelectC += cantSelect
		maxIdxTimeC += maxIdxTime 
		fpSelectTimeC += fpSelectTime 
		idTimeC += idTime	


		if i == ganador:
			aciertos[i] = 1

	tiempos = np.array([photoTime, patchesTime, adaptiveTime, fpTime, maxIdxTimeC, fpSelectTimeC, idTimeC, cantSelectC])

	return aciertos, tiempos


def testingDoble(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, s, ii, jj, L, w, alpha, sub, neigh, 
	sparseThreshold, SCIThreshold, distThreshold, useAlpha, tanTriggs):
	# testing kNNSparse_v1.1 OFICIAL
	cantPersonas = len(idxPerson)
	idxTestPhoto = idxPhoto.shape[1]-1
	aciertos = np.zeros(cantPersonas)
	aciertosAlt = np.zeros(cantPersonas)


	for i in range(cantPersonas):
		# Ruta de la foto de testing
		inicio = time.time()
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		I = imageUtils.readScaleImage(routePhoto, width, height, tanTriggs=tanTriggs) # lectura de la imagne
		
		inicio = time.time()
		alpha1 = magisterUtils.fingerprint(I, U, YC, ii, jj, L, w, alpha, sub, neigh, distThreshold, useAlpha=useAlpha)
		alpha1 = alpha1.transpose()
		ganador = clasifier(alpha1, Q, R, s, L, SCIThreshold, sparseThreshold, cantPersonas)


		
		alpha2 = magisterUtils.fingerprintOld(I, U, YC, ii, jj, L, w, alpha, sub, useAlpha=useAlpha)
		alpha2 = alpha2.transpose()
		ganadorAlt = clasifier(alpha2,  Q, R, s, L, SCIThreshold, sparseThreshold, cantPersonas)
		

		if i == ganador:
			aciertos[i] = 1

		if i == ganadorAlt:
			aciertosAlt[i] = 1
	
	return aciertos, aciertosAlt


def testingLFWa(dataBasePath, idxPerson, width, height, U, YC, Q, R, s, ii, jj, L, w, alpha, sub, 
	sparseThreshold, SCIThreshold, useAlpha, tanTriggs):
	
	cantPersonas = len(idxPerson)
	aciertos = np.zeros(0)
	contExp = 0.0	
	for i, person in enumerate(idxPerson):
		allPhotos = dataBaseUtils.totalPhotos(dataBasePath, person)
		testPhotos = allPhotos[10:]

		route = os.path.join(dataBasePath, person)
		
		for photo in testPhotos:
			routePhoto = os.path.join(route, photo)
			I = imageUtils.readScaleImage(routePhoto, width, height, tanTriggs=tanTriggs) # lectura de la imagne
			
			alpha1 = magisterUtils.fingerprint(I, U, YC, ii, jj, L, w, alpha, sub, useAlpha=useAlpha)
			alpha1 = alpha1.transpose()
			
			ganador = clasifier(alpha1, Q, R, s, L, SCIThreshold, sparseThreshold, cantPersonas)
			
			if i == ganador:
				aciertos = np.append(aciertos,1)
			else:
				aciertos = np.append(aciertos,0)

			contExp += 1
			print "Porcentaje Aciertos: " , float(aciertos.sum())/contExp*100, "%"	
			
	return aciertos


def clasifier(alpha1, Q, R, s, L, SCIThreshold, sparseThreshold, cantPersonas):	
	
	
	inicio = time.time()
	alphaMax = magisterUtils.alphaMaxIdxSelect(alpha1)
	maxIdxTime = time.time()-inicio

	inicio = time.time()
	alphaFinal = magisterUtils.fingerprintSelect(alpha1, alphaMax, s, Q, R, cantPersonas, L, SCIThreshold)
	fpSelectTime = time.time()-inicio

	

	inicio = time.time()
	correcto = identitySelection(alphaFinal, Q, R, cantPersonas)
	idTime = time.time()-inicio

	return correcto, maxIdxTime, fpSelectTime, idTime, alphaFinal.shape[0]

def clasifierAdaptive(alpha1, Q, R, s, L, SCIThreshold, sparseThreshold, cantPersonas):	
	
	alphaMax = magisterUtils.alphaMaxIdxSelect(alpha1)
	alphaFinal = magisterUtils.fingerprintSelect(alpha1, alphaMax, s, Q, R, cantPersonas, L, SCIThreshold)

	correcto = identitySelectionAdaptive(alphaFinal, R, cantPersonas)
	return correcto

def clasifierAlt(alpha1, Q, R, L, SCIThreshold, sparseThreshold, cantPersonas):	
	
	alphaMax = magisterUtils.alphaMaxIdxSelect(alpha1)
	# validIdx, sciVector = magisterUtils.validIndex_v2(alpha1, Q, R, cantPersonas, L, SCIThreshold)
	# alphaFinal = alphaMax[validIdx,:]
	
	correcto = identitySelection(alphaMax, Q, R, cantPersonas)
	return correcto



def identitySelection(alphaFinal, Q, R, cantPersonas):

	alphaSum = np.sum(alphaFinal,axis=0)
	maximo = 0
	
	for j in range(cantPersonas):
		
		alphaAux = alphaSum[j*Q*R:(j+1)*Q*R]
		
		suma = np.sum(alphaAux)
		
		if suma > maximo:
			maximo = suma
			correcto = j

	return correcto


def identitySelectionAdaptive(alphaFinal, R, cantPersonas):

	alphaSum = np.sum(alphaFinal,axis=0)
	maximo = 0
	
	for j in range(cantPersonas):
		
		alphaAux = alphaSum[j*R:(j+1)*R]
		
		suma = np.sum(alphaAux)
		
		if suma > maximo:
			maximo = suma
			correcto = j

	return correcto

def generateExperiment(dataBasePath, cantPhotosPerPerson, cantPersonas, cantPhotosDict, cantIteraciones):
	# Genera cantIteraciones de experimentos de una, para correr distintos parametros con lo mismo y hacerlo comparable
	idxPersonFull = np.array([])
	idxPhotoFull = np.array([])
	cantPhotos = cantPhotosDict+1

	for i in range(cantIteraciones):
		idxPerson, idxPhoto = dataBaseUtils.randomSelection_Old(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas)
		idxPersonFull = miscUtils.concatenate(idxPerson, idxPersonFull, 'vertical')
		idxPhotoFull = miscUtils.concatenate(idxPhoto, idxPhotoFull, 'vertical')



	return idxPersonFull, idxPhotoFull

