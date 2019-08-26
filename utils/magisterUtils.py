# -*- coding: utf-8 -*-
"""
Librería para algoritmo Magister
Tomás Larrain A.
"""

import cv2
import numpy as np
import spams
import miscUtils
import imageUtils
import ASRUtils as asr
from scipy import signal
from scipy.spatial import distance as dist
from sklearn.neighbors import KNeighborsClassifier
from scipy import io
import os
import time

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


def fingerprintDist(Y, YC,  L, neigh, distThreshold, useAlpha=True, tipo='omp'):
	# Generación de fingerprint sparse de la imagen I	
	

	dist, win = neigh.kneighbors(Y, 1, return_distance=True)
	dist = dist.flatten()
	Y = patchSelect(Y, dist, distThreshold)
	
	if tipo == 'omp':
		inicio = time.time()
		alpha1 = asr.normL1_omp(Y, YC, L)	
		
		return alpha1

	if tipo == 'lasso':
		alpha1 = asr.normL1_lasso(Y, YC, L)
		# alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1), order='F')
		return alpha1

	else:
		print "Tipo de minimización no válido"
		exit()	


def fingerprint(Y, YC, L, tipo='omp'):
	# Generación de fingerprint sparse de la imagen I	
	if tipo == 'omp':
		inicio = time.time()
		alpha1 = asr.normL1_omp(Y, YC, L)	
		# print '\tOMP: ', time.time()-inicio, ' segundos'
		# alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1))
		# alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1), order='F')
		return alpha1

	if tipo == 'lasso':
		alpha1 = asr.normL1_lasso(Y, YC, L)
		# alpha1 = np.reshape(alpha1, (alpha1.shape[0]*alpha1.shape[1], 1), order='F')
		return alpha1

	else:
		print "Tipo de minimización no válido"
		exit()	


def fingerprintAdaptive(Y, YC, LimMat, L, R, cantPersonas, neigh = [], distThreshold = 10, selPatches = False):
	# Generación de fingerprint sparse de la imagen I
	
	if selPatches:
		dist, win = neigh.kneighbors(Y, 1, return_distance=True)
		dist = dist.flatten()
		Y = patchSelect(Y, dist, distThreshold)
	
	
	n = Y.shape[0]
	
	alphas = np.zeros((n,R*cantPersonas))

	for i in range(n):
		patch = Y[i,:]
		Dict = YC[LimMat[i,:],:]

		inicio = time.time()
		alpha1 = asr.normL1_omp(patch, Dict, L)	
		alphas[i,:] = alpha1.transpose()


	return alphas


def generateQueryBase(dataBasePath, idxPerson, idxPhoto, cantPhotosSparse, U, YC, iiSparse, jjSparse, L, width, height, w, alpha, sub, 
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
			I = imageUtils.readScaleImage(routePhoto, width, height)
			
			alpha1 = fingerprint(I, U, YC, iiSparse, jjSparse, L, w, alpha, sub, useAlpha)
			Ysparse = miscUtils.concatenate(alpha1, Ysparse, 'horizontal')
		
	Ysparse = Ysparse.transpose()
	
	Ysparse = (Ysparse < -sparseThreshold) | (Ysparse > sparseThreshold) # por umbral
	# Ysparse = Ysparse != 0
	return Ysparse


def binarize(alpha1, sparseThreshold):
	# binariza de acuerdo al umbral
	return (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold)


def validMatrix(alphaR, Q, R, cantPersonas, L, SCIThreshold):
	cantSparse = alphaR.shape[0]
	valid = np.zeros(alphaR.shape)
	
	if SCIThreshold == 0:
		return valid+1

	for i in range(cantSparse):
		alpha = alphaR[i,:]
		sci = asr.SCI(alpha, Q, R, cantPersonas, L)
		
		if sci > SCIThreshold:
			valid[i,:] = 1

	return valid

def validIndex(alphaR, Q, R, cantPersonas, L, SCIThreshold):
	cantSparse = alphaR.shape[0]
	valid = np.zeros(cantSparse)

	if SCIThreshold == 0 or L == 1:
		return np.array(range(cantSparse))

	cont = 0	
	for i in range(cantSparse):
		alpha = alphaR[i,:]
		
		sci = asr.SCI(alpha, Q, R, cantPersonas, L)
	
		if sci > SCIThreshold:
			valid[cont] = i
			cont += 1


	return valid[:cont].astype(int)

def alphaMaxIdxSelect(alpha1):
	
	idx = np.argmax(alpha1,axis=1)
	alphaMax = np.zeros(alpha1.shape)
	
	for i in range(alphaMax.shape[0]):
		alphaMax[i,idx[i]] = 1

	return alphaMax


def validIndex_v2(alphaR, Q, R, cantPersonas, L, SCIThreshold):
	# Selecciona los indices de los parches que tienen un SCI > SCIThreshold
	cantSparse = alphaR.shape[0]
	valid = np.zeros(cantSparse)
	sciVector = np.zeros(cantSparse)
	
	if L == 1:
		return np.array(range(cantSparse)), int(0) 

	cont = 0	
	for i in range(cantSparse):
		alpha = alphaR[i,:]
		
		sci = asr.SCI(alpha, Q, R, cantPersonas, L)
	
		if sci > SCIThreshold:
			valid[cont] = i
			sciVector[cont] = sci
			cont += 1

	valid = valid[:cont].astype(int)
	sciVector = sciVector[:cont].astype(int)

	return valid, sciVector


def bestPatches(valid, sciVector, s):
	# Selecciona los s mejores parches, post filtro SCI
	idx = np.argsort(sciVector)
	
	if type(sciVector) == int:
		return valid

	valid = valid[idx]
	valid = valid[-s:]
	
	return valid


def patchSelect(Y, dist, distThreshold):
	# Selecciona los parches con cercanía mayor a distThreshold al diccionario
	boolean = dist<= distThreshold
	Y = Y[boolean]
	return Y

def fingerprintSelect(alpha1, alphaMax, s, Q, R, cantPersonas, L, SCIThreshold):
	# Extrate de alpha1 los mejores parches despues del filtro del SCI y de s
	validIdx, sciVector = validIndex_v2(alpha1, Q, R, cantPersonas, L, SCIThreshold)
	validIdx = bestPatches(validIdx, sciVector, s)
	alphaFinal = alphaMax[validIdx,:]

	return alphaFinal


