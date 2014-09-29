# -*- coding: utf-8 -*-
"""
Miscelaneous Utils
Tomás Larrain A.
4 de agosto de 2014
"""
import cv2
import os
import numpy as np


def responseVector(cantPersonas, idxPerson, cantPhotosSparse):
	# Vector de representación ideal  con cantPersonas personas que tienen cantPhotosSparse fotos
	responses = np.zeros(0)
	for i in range(cantPersonas): 
		responses = np.append(responses,float(idxPerson[i])*np.ones(cantPhotosSparse))
	
	return responses


def concatenate(auxMatrix, acumMatrix, direction):
		
	if len(acumMatrix) == 0:
		acumMatrix = auxMatrix.copy()
	
	elif direction == 'vertical':
		acumMatrix = np.vstack((acumMatrix,auxMatrix))

	elif direction == 'horizontal':
		acumMatrix = np.hstack((acumMatrix,auxMatrix))	
	
	return acumMatrix


def readScaleImageBW(route, width, height):
	# Lee la imagen de route, la pasa a B&N y la escala segun width y height
	It = cv2.imread(route)
	if It is not None:	 
		It = cv2.cvtColor(It,cv2.COLOR_BGR2GRAY)
		It = np.float32(It)
		It = cv2.resize(It,(width,height))	
		return It
	else:
		return np.zeros(0)


def readScaleImageColor(route, width, height):
	# Lee la imagen de route, la pasa a B&N y la escala segun width y height
	It = cv2.imread(route)
	if It is not None:	 
		It = np.float32(It)
		It = cv2.resize(It,(width,height))	
		return It
	else:
		return np.zeros(0)


def wiener(I):
	# Filtro de Wiener para imágenes
	uni = I/255.0
	W = signal.wiener(uni, mysize=(3,3))
	W = np.round(255*W)
	W = np.uint8(W)
	return W


def fixedLengthString(longString, shortString):
	diff = len(longString) - len(shortString)
	outString = ' '*diff
	outString += shortString
	return outString	


def returnUnique(array):
	arrayUnique, idx = np.unique(array,return_index = True)		
	idx = np.sort(idx)
	return array[idx]	

