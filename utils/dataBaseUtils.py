# -*- coding: utf-8 -*-
"""
Data Base Utils
Tomás Larrain A.
4 de agosto de 2014
"""
import cv2
import os
import numpy as np
import utils.miscUtils as miscUtils

def getFacePath():
	# Path donde estan todas las bases de datos
	return '/Users/Tomas/Developer/data/faces/'


def getDataBasePath(dataBase):
	# Retorna el path donde se encuentra la base de datos dataBase
	
	facePath = getFacePath()

	if dataBase == 'AR' or dataBase == 'ARx':
		return os.path.join(facePath,"AR/CROP/"), 26

	if dataBase == 'ORL':
		return os.path.join(facePath,"ORL/NOCROP/"), 10

	if dataBase == 'Nutrimento':
		return os.path.join(facePath,"Nutrimento/CROP/"), 0

	if dataBase == 'Junji':
		return os.path.join(facePath,"Junji/CROP/"), 0

	if dataBase == 'LFW':
		return os.path.join(facePath, "LFW"), 0

	if dataBase == 'Yale':
		return os.path.join(facePath, "Yale"), 59

	if dataBase == 'FWM':
		return os.path.join(facePath, "FWM"), 0

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


def personSelectionByPhotoAmount(dataBasePath, photoAmount):
	# Retorna los ID de las personas que contienen >= cantidad de fotos que photoAmount

	idxPerson = getPersonIDs(dataBasePath)
	personSelection = np.array([])
	for d in range(len(idxPerson)):

		if idxPerson[d] == '.DS_Store':
			continue

		dataBase = os.path.join(dataBasePath,idxPerson[d])
		files = os.listdir(dataBase)

		if len(files)>=photoAmount:
			personSelection = np.append(personSelection,idxPerson[d])

	return personSelection


def randomPersonSelection(dataBasePath, idxPerson, cantPersonas):
	# Selección de fotos y de personas aleatorio

	
	auxIdx = np.random.permutation(len(idxPerson))[:cantPersonas]
	
	return idxPerson[auxIdx]


def randomPhotoSelection(dataBasePath, idxPerson, cantPhotos):
	# Selección de fotos aleatorias por persona
	idxPhoto = np.array([])

	for i in range(len(idxPerson)):
		photos = totalPhotos(dataBasePath, idxPerson[i])
		idxPhoto = miscUtils.concatenate(np.random.permutation(photos)[:cantPhotos],idxPhoto,'vertical')
	
	return idxPhoto

def randomSelection(dataBasePath, idxPerson, cantPhotos, cantPersonas):
	
	auxIdx = np.random.permutation(len(idxPerson))[:cantPersonas]
	idxPerson = idxPerson[auxIdx]

	idxPhoto = np.array([])

	for i in range(len(idxPerson)):
		photos = totalPhotos(dataBasePath, idxPerson[i])
		idxPhoto = miscUtils.concatenate(np.random.permutation(photos)[:cantPhotos],idxPhoto,'vertical')
	
	return idxPerson, idxPhoto

def totalPhotos(dataBasePath, idxSinglePerson):
	# Cantidad de fotos de una sola persona 
	personPath = os.path.join(dataBasePath, idxSinglePerson)
	totalPhotos = len(os.listdir(personPath))
	
	return totalPhotos


def randomSelection_Old(dataBasePath, cantPhotosPerPerson, cantPhotos, cantPersonas):
	# Selección de fotos y de personas aleatorio

	idxPerson = getPersonIDs(dataBasePath)
	auxIdx = np.random.permutation(len(idxPerson))[:cantPersonas]
	
	idxPerson = idxPerson[auxIdx]
	idxPhoto = np.random.permutation(cantPhotosPerPerson)[:cantPhotos]
	
	idxPhoto = np.tile(idxPhoto,(cantPersonas,1))

	return idxPerson, idxPhoto

def randomSelectionPhoto_ARx(cantPhotos, cantPersonas, tipo='old'):
	# Seleccion de fotos para ARx
	trainSet = np.array([1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20]) - 1 
	testSet = np.array([8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26]) - 1
	idxPhoto = np.array([])

	if tipo == 'old':
		
		idxPhotoTrain = np.random.permutation(trainSet)[:cantPhotos-1]
		idxPhotoTest = np.random.choice(testSet, 1)[0]

		idxPhoto = np.append(idxPhotoTrain, idxPhotoTest)
		idxPhoto = np.tile(idxPhoto,(cantPersonas, 1))

		return idxPhoto

	if tipo == 'new':

		for i in range(cantPersonas):
			idxPhotoTrain = np.random.permutation(trainSet)[:cantPhotos-1]
			idxPhotoTest = np.random.choice(testSet, 1)[0]
			idxPhotoAux = np.append(idxPhotoTrain, idxPhotoTest)
			idxPhoto = miscUtils.concatenate(idxPhotoAux, idxPhoto, 'vertical')

		return idxPhoto



