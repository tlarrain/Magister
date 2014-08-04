# -*- coding: utf-8 -*-
"""
Miscelaneous Utils
Tomás Larrain A.
4 de agosto de 2014
"""
import os


def getFacePath():
	return '/Users/Tomas/Developer/data/faces/'


def getDataBasePath(dataBase):

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


def photosPerPerson(rootPath):
	# cantidad minima de fotos existentes en una clase (para que todo esten iguales)
	idx_person = os.listdir(rootPath)
	cant_fotos = float('inf')
	for d in range(len(idx_person)):
		if idx_person[d] == '.DS_Store':
			continue
		root = os.path.join(rootPath,idx_person[d])
		files = os.listdir(root)
		if len(files)<cant_fotos:
			cant_fotos = len(files)
	return cant_fotos			


def randomSelection(rootPath,cantFotos,cantPersonas):
	
	idxPerson = np.array([d for d in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, d))])
	auxIdx = np.random.permutation(len(idxPerson))[:cantPersonas]
	idxPerson = idxPerson[auxIdx]
	sujetos = len(idxPerson)
	idxFoto = np.random.permutation(cantFotos)


def drawPatch(I,corner,a,b):
	# Dibuja un patch de tamanio (a,b) desde la esquina corner en la imagen I.
	x = int(corner[1])
	y = int(corner[0])
	cv2.rectangle(I,(x,y),(x+a,y+b),(255,0,0))
	return I


