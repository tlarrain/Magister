# -*- coding: utf-8 -*-
"""
Figuras y visualizaciones
Tomás Larrain A.
13 de Agosto de 2014
"""

import numpy as np
import ASRUtils as asr
import miscUtils
import imageUtils
import cv2
import os

def drawPatch(I, corner, a, b, blue, green, red):
	# Dibuja un patch de tamanio (a,b) desde la esquina corner en la imagen I.
	
	x = int(corner[1])
	y = int(corner[0])
	cv2.rectangle(I,(x,y),(x+a-1,y+b-1),(blue,green,red))
	
	return I


def displayGrilla(I,ii,jj,a,b,m):
	# Dibuja la grilla en colores intercalados
	for i in range(m):
		
		if i%2 == 0:
			I = drawPatch(I,(ii[i],jj[i]),a,b, 255, 255, 0)
		
		else:
			I = drawPatch(I,(ii[i],jj[i]),a,b, 0, 255, 255)

	cv2.namedWindow("Grilla")
	cv2.imshow("Grilla",np.uint8(I))
	cv2.waitKey()

	return np.uint8(I)



def generateAllPhotos(cantPersonas, cantPhotosDict, idxPhoto, idxPerson, rootPath, dispWidth, dispHeight, aciertos):

	displayImage = np.array([])
	blackSpace = np.zeros((dispHeight,20,3)) # espacio para separar fotos del diccionario de base de datos

	for i in range(cantPersonas):
		fila = np.array([])
		route = os.path.join(rootPath, idxPerson[i])
		photos = os.listdir(route)

		for d in range(cantPhotosDict):
			routePhoto = os.path.join(route, photos[idxPhoto[i,d]]) # ruta de la foto j
			I = imageUtils.readScaleImage(routePhoto, dispWidth, dispHeight,tipo='color') # lectura de la imagen
			fila = miscUtils.concatenate(I, fila, 'horizontal')

		fila = miscUtils.concatenate(blackSpace, fila, 'horizontal')
		routePhoto = os.path.join(route, photos[idxPhoto[i,cantPhotosDict]]) # ruta de la foto de test
		I = imageUtils.readScaleImage(routePhoto, dispWidth, dispHeight,tipo='color') # lectura de la imagen de test
		
		if aciertos[i] == 0:
			I = drawPatch(I, (0,0), dispWidth, dispHeight, 0, 0, 255)

		else: 
			I = drawPatch(I, (0,0), dispWidth, dispHeight, 0, 255, 0)

		fila = miscUtils.concatenate(I, fila, 'horizontal')		
		displayImage = miscUtils.concatenate(fila, displayImage, 'vertical')


	return np.uint8(displayImage)

def displayResults(results, allPhotos):
	# Función que despliega las imagenes en una sola ventana
	displayImage = np.array([])
	blackSpace = np.zeros((results.shape[0],20,3))
	blackSpace = np.uint8(blackSpace)
	displayImage = miscUtils.concatenate(blackSpace, allPhotos, 'horizontal')
	displayImage = miscUtils.concatenate(results,displayImage, 'horizontal')

	cv2.namedWindow("Resultados finales")
	cv2.imshow("Resultados finales", displayImage)
	cv2.waitKey()


#######################################


def generateResults(correctPhoto, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, rootPath, dispWidth, dispHeight):	
	# Desplega la imagen query y aquella a la que encontró más parecida. Para continuar de presiona cualquier tecla
	cantPersonas = correctPhoto.shape[0]
	
	displayImage = np.array([])
	idxTestPhoto = idxPhoto.shape[1]-1

	for i in range(cantPersonas):
		possibleMatchPhotos = idxPhoto[i,0:cantPhotosSparse+1]
		matchPhoto = correctPhoto[i,0]
		matchPhoto = int(matchPhoto%cantPhotosSparse)
		matchPhoto = possibleMatchPhotos[matchPhoto]
		matchPerson = int(correctPhoto[i,0])/int(cantPhotosSparse)
		matchPerson = idxPerson[matchPerson]
		
		queryPerson = idxPerson[i]
		queryPhoto = idxPhoto[i,idxTestPhoto]
		
		queryRoute = os.path.join(rootPath, queryPerson)
		queryPhotos = os.listdir(queryRoute)
		queryRoutePhoto = os.path.join(queryRoute, queryPhotos[queryPhoto])

		matchRoute = os.path.join(rootPath, matchPerson)
		matchPhotos = os.listdir(matchRoute)
		matchRoutePhoto = os.path.join(matchRoute, matchPhotos[matchPhoto])

		Iq = imageUtils.readScaleImage(queryRoutePhoto, dispWidth, dispHeight, tipo='color')
		Im = imageUtils.readScaleImage(matchRoutePhoto, dispWidth, dispHeight, tipo='color')

		if correctPhoto[i,1] == 0:
			Im = drawPatch(Im, (0,0), dispWidth, dispHeight, 0, 0, 255)

		else:
			Im = drawPatch(Im, (0,0), dispWidth, dispHeight, 0, 255, 0)			

		fila = np.hstack((Iq,Im))
		displayImage = miscUtils.concatenate(fila, displayImage, 'vertical')

	return np.uint8(displayImage)
	
