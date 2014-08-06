# -*- coding: utf-8 -*-
"""
Miscelaneous Utils
Tomás Larrain A.
4 de agosto de 2014
"""
import cv2
import os
import numpy as np

def getFacePath():
	# Path donde estan todas las bases de datos
	return '/Users/Tomas/Developer/data/faces/'


def getDataBasePath(dataBase):
	# Retorna el path donde se encuentra la base de datos dataBase
	
	facePath = getFacePath()

	if dataBase == 'AR':
		return os.path.join(facePath,"AR/CROP/")

	if dataBase == 'ORL':
		return os.path.join(facePath,"ORL/NOCROP/")

	if dataBase == 'Nutrimento':
		return os.path.join(facePath,"Nutrimento/CROP/")

	if dataBase == 'Junji':
		return os.path.join(facePath,"Junji/CROP/")

	else:
		return "No data base with " + str(dataBase) + " name in the face path"


def getPersonIDs(dataBasePath):
	# Retorna el ID de las personas del dataBasePath
	return np.array([d for d in os.listdir(dataBasePath) if os.path.isdir(os.path.join(dataBasePath, d))])


def photosPerPerson(dataBasePath):
	# Cantidad minima de fotos existentes en una clase (para que todo esten iguales)
	
	idxPerson = os.listdir(dataBasePath)
	cantFotos = float('inf')
	
	for d in range(len(idxPerson)):
		if idxPerson[d] == '.DS_Store':
			continue
		dataBase = os.path.join(dataBasePath,idxPerson[d])
		files = os.listdir(dataBase)
		
		if len(files)<cantFotos:
			cantFotos = len(files)
	
	return cantFotos			


def randomSelection(dataBasePath, cantFotos, cantPersonas):
	# Selección de fotos y de personas aleatorio

	idxPerson = getPersonIDs(dataBasePath)
	auxIdx = np.random.permutation(len(idxPerson))[:cantPersonas]
	
	idxPerson = idxPerson[auxIdx]
	idxFoto = np.random.permutation(cantFotos)
	
	return idxFoto, idxPerson


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


def readScaleImage(route, width, height):
	# Lee la imagen de route, la pasa a B&N y la escala segun width y height
	It = cv2.imread(route)
	if It is not None:	 
		It = cv2.cvtColor(It,cv2.COLOR_BGR2GRAY)
		It = np.float32(It)
		It = cv2.resize(It,(width,height))	
		return It
	else:
		return np.zeros(0)


def drawPatch(I, corner, a, b):
	# Dibuja un patch de tamanio (a,b) desde la esquina corner en la imagen I.
	
	x = int(corner[1])
	y = int(corner[0])
	cv2.rectangle(I,(x,y),(x+a,y+b),(255,0,0))
	
	return I


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

