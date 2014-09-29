# -*- coding: utf-8 -*-
"""
Librería para algoritmo Magister
Tomás Larrain A.
"""

import cv2
import numpy as np
import spams
import miscUtils
import ASRUtils as asr
from scipy import signal
from scipy.spatial import distance as dist
from sklearn.neighbors import KNeighborsClassifier
from scipy import io
import os


def distance(array1, array2, tipo):
	# Distintos tipos de métricas de distancia
	eps = 1e-5
	length = len(array1)
	
	if tipo == 'euclidean':
		return	dist.euclidean(array1,array2)

	if tipo == 'hamming':
		return dist.hamming(array1,array2)*length

	if tipo == 'absDiff':
		return np.sum(np.abs(array1-array2),axis=1)
	
	if tipo == 'chiSquare':
		return np.sum(0.5*np.divide(((array1-array2)**2),(array1+array2+eps)))
	else:
		print "Tipo de distancia no válido"
		exit()


def fingerprint(I, U, YC, ii, jj, L, a, b, alpha, sub, useAlpha=True, tipo='omp'):
	# Generación de fingerprint sparse de la imagen I	
	height = I.shape[0]
	width = I.shape[1]
	Y = asr.patches(I, ii, jj, U, a, b, alpha, sub, useAlpha)
	
	# np.save("YC.npy", YC)
	# np.save("patches.npy", Y)

	# io.savemat("/Users/Tomas/Developer/Matlab/Magister/variables.mat",{'YC':YC, 'patches':Y})

	if tipo == 'omp':
		alpha1 = asr.normL1_omp(Y, YC, L)
		# alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1))
		# alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1), order='F')
		return alpha1

	if tipo == 'lasso':
		alpha1 = asr.normL1_lasso(Y, YC, L)
		alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1), order='F')
		return alpha1

	else:
		print "Tipo de minimización no válido"
		exit()	


def generateQueryBase(dataBasePath, idxPerson, idxPhoto, cantPhotosSparse, U, YC, iiSparse, jjSparse, L, width, height, a, b, alpha, sub, 
	useAlpha, sparseThreshold, distType):
	
	Ysparse = np.array([])
	cantPersonas = len(idxPerson)
	
	for i in range(cantPersonas):
	
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		
		for j in range(cantPhotosSparse):
			# idx = j+cantPhotosDict
			idx = j
			routePhoto = os.path.join(route, photos[idxPhoto[i,idx]])
			I = miscUtils.readScaleImageBW(routePhoto, width, height)
			
			alpha1 = fingerprint(I, U, YC, iiSparse, jjSparse, L, a, b, alpha, sub, useAlpha)
			Ysparse = miscUtils.concatenate(alpha1, Ysparse, 'horizontal')
		
	Ysparse = Ysparse.transpose()
	
	Ysparse = (Ysparse < -sparseThreshold) | (Ysparse > sparseThreshold) # por umbral
	# Ysparse = Ysparse != 0
	return Ysparse


def clasifier(Ysparse, alpha1, responses, distType):
	# Clasificador original del algoritmo
	total = Ysparse.shape[0]
	resto = float('inf')

	for j in range(total):
			
		Yclass = Ysparse[j, :] # matriz sparse que representa la foto	
		restoAux = distance(Yclass,alpha1,distType) # valor absoluto de la resta
		# Encuentra la resta con menor error
		if restoAux < resto:
			correctPhoto = j
			correctID = responses[j]
			resto = restoAux

	return correctPhoto, correctID


def clasifier_v2(alpha1, Q, R, m, sparseThreshold, cantPersonas):
	
	alpha1 = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold)
	alpha1 = alpha1.flatten()
	alphaTile = np.reshape(alpha1,(cantPersonas,Q*R*m))
	suma = np.sum(alphaTile, axis=1)
	ganador = np.argmax(suma)

	return ganador


def clasifier_v3(alpha1, Q, R, m, L, SCIThreshold, sparseThreshold, cantPersonas):
	
	# alphaBinary = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold)
	# alpha1 = alpha1.flatten()
	# alphaR = np.reshape(alpha1,(m, Q*R*cantPersonas))
	
	# Extrae solamente el coeficiente sparse mas alto de cada fila
	idx = np.argmax(alpha1,axis=1)
	alphaMax = np.zeros(alpha1.shape)
	for i in range(alphaMax.shape[0]):
		alphaMax[i,idx[i]] = 1

	valid = validMatrix(alpha1, Q, R, cantPersonas, L, SCIThreshold)
	alphaFinal = alphaMax*valid
	# print (np.nonzero(alphaFinal != 0))[0].shape # muestra la cantidad de patches que pasaron la prueba
	
	maximo = 0
	for j in range(cantPersonas):
		
		alphaAux = alphaFinal[:,j*Q*R:(j+1)*Q*R]
		suma = alphaAux.sum()
		
		if suma > maximo:
			maximo = suma
			correcto = j


	return correcto


def testing(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Ysparse, ii, jj, L, a, b, alpha, sub, 
	sparseThreshold, useAlpha, distType, responses):
	# testing algoritmo
	cantPersonas = idxPhoto.shape[0]
	idxTestPhoto = idxPhoto.shape[1]-1
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior
	aciertos = 0

	for i in range(cantPersonas):
	# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, L, a, b, alpha, sub, useAlpha)
		
		# Inicialización variables de testing
		resto = float('inf')
		corrPhoto = cantPersonas+1
		
		# Binarización representaciones sparse
		alpha1 = alpha1.transpose()
		alphaBinary = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral
		

		corrPhoto, corrID = clasifier(Ysparse, alphaBinary, responses, distType)
		correctPhoto[i,0] = corrPhoto

		
		# Compara con vector de clasificación ideal
		if int(corrID) == int(idxPerson[i]):
			aciertos += 1
			correctPhoto[i,1] = 1

	return aciertos, correctPhoto	


def testing_v2(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, m, ii, jj, L, a, b, alpha, sub, 
	sparseThreshold, useAlpha):
	# testing kNNSparse_v1.1
	aciertos = 0
	cantPersonas = len(idxPerson)
	idxTestPhoto = idxPhoto.shape[1]-1
	# Ytest = np.array([])
	
	for i in range(cantPersonas):
		# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, L, a, b, alpha, sub, useAlpha)
		ganador = clasifier_v2(alpha1, Q, R, m, sparseThreshold, cantPersonas)

		# Ytest = miscUtils.concatenate(alpha1,Ytest,'horizontal')

		if i == ganador:
			aciertos += 1

	# Ytest = Ytest.transpose()
	# io.savemat("/Users/Tomas/Developer/Matlab/Magister/Ysparse.mat",{'Ysparse':Ytest})
	# np.save("Ysparse.npy",Ytest)
	
	return aciertos


def testing_v3(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Q, R, m, ii, jj, L, a, b, alpha, sub, 
	sparseThreshold, SCIThreshold, useAlpha):
	# testing kNNSparse_v1.1
	aciertos = 0
	cantPersonas = len(idxPerson)
	idxTestPhoto = idxPhoto.shape[1]-1
	# Ytest = np.array([])
	for i in range(cantPersonas):
		# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, L, a, b, alpha, sub, useAlpha)
		
		# Ytest = miscUtils.concatenate(alpha1,Ytest,'horizontal')
		alpha1 = alpha1.transpose()
		ganador = clasifier_v3(alpha1, Q, R, m, L, SCIThreshold, sparseThreshold, cantPersonas)

		if i == ganador:
			aciertos += 1

	
	return aciertos


def testing_Scikit(dataBasePath, idxPerson, idxPhoto, n_neighbors, width, height, U, YC, Ysparse, ii, jj, L, a, b, alpha, sub, 
	sparseThreshold, useAlpha, distType, responses):
	# Testing co scikit y alteatoriedad irregular
	cantPersonas = idxPhoto.shape[0]
	idxTestPhoto = idxPhoto.shape[1]-1
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior
	aciertos = np.zeros(len(n_neighbors))

	Ytest = np.array([])

	for i in range(cantPersonas):
		# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, L, a, b, alpha, sub, useAlpha)
		
		alpha1 = alpha1.transpose()
		# Binarización representaciones sparse
		
		Ytest = miscUtils.concatenate(alpha1, Ytest, 'vertical')
	

	Ytest = (Ytest < -sparseThreshold) | (Ytest > sparseThreshold) # por umbral

	for n in range(len(n_neighbors)):
		neigh = KNeighborsClassifier(n_neighbors=n_neighbors[n],metric=distType) # inicializacion clasificador
		neigh.fit(Ysparse, responses) # entrenador del algoritmo
		resultIDs = neigh.predict(Ytest) # testing
		resultPhotos = neigh.kneighbors(Ytest, n_neighbors=n_neighbors[n], return_distance=False) # foto más cercana
			
	
		for i in range(cantPersonas):

			correctPhoto[i,0] = resultPhotos[i][0]	
			
			# Compara con vector de clasificación ideal
			if int(resultIDs[i]) == int(idxPerson[i]):
				aciertos[n] += 1
				correctPhoto[i,1] = 1

	return aciertos, correctPhoto


def validMatrix(alphaR, Q, R, cantPersonas, L, SCIThreshold):
	cantSparse = alphaR.shape[0]
	valid = np.zeros(alphaR.shape)
	
	for i in range(cantSparse):
		alpha = alphaR[i,:]
		sci = asr.SCI(alpha, Q, R, cantPersonas, L)
		
		if sci > SCIThreshold:
			valid[i,:] = 1

	return valid






