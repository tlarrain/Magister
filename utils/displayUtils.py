# -*- coding: utf-8 -*-
"""
Figuras y visualizaciones
Tomás Larrain A.
13 de Agosto de 2014
"""

import numpy as np
import utils.ASRUtils as asr
import utils.miscUtils as miscUtils
import cv2
import os

def displayGrilla(I,ii,jj,a,b,m):
	# Dibuja la grilla en colores intercalados
	for i in range(m):
		
		if i%2 == 0:
			I = miscUtils.drawPatch(I,(ii[i],jj[i]),a,b, 255, 255, 0)
		
		else:
			I = miscUtils.drawPatch(I,(ii[i],jj[i]),a,b, 0, 255, 255)

	cv2.namedWindow("Grilla")
	cv2.imshow("Grilla",np.uint8(I))
	cv2.waitKey()


def displayResults(correctPhoto, cantPhotosDict, cantPhotosSparse, idxPhoto, idxPerson, rootPath, dispWidth, dispHeight):	
	# Desplega la imagen query y aquella a la que encontró más parecida. Para continuar de presiona cualquier tecla
	cantPersonas = len(correctPhoto)
	possibleMatchPhotos = idxPhoto[cantPhotosDict:cantPhotosSparse+1]
	displayImage = np.array([])
	idxTestPhoto = len(idxPhoto)-1

	for i in range(cantPersonas):
		
		matchPhoto = correctPhoto[i]
		matchPhoto = int(matchPhoto%cantPhotosSparse)
		matchPhoto = possibleMatchPhotos[matchPhoto]
		matchPerson = int(correctPhoto[i])/int(cantPhotosSparse)
		matchPerson = idxPerson[matchPerson]
		
		queryPerson = idxPerson[i]
		queryPhoto = idxPhoto[idxTestPhoto]
		
		queryRoute = os.path.join(rootPath, queryPerson)
		queryPhotos = os.listdir(queryRoute)
		queryRoutePhoto = os.path.join(queryRoute, queryPhotos[queryPhoto])

		matchRoute = os.path.join(rootPath, matchPerson)
		matchPhotos = os.listdir(matchRoute)
		matchRoutePhoto = os.path.join(matchRoute, matchPhotos[matchPhoto])

		Iq = miscUtils.readScaleImage(queryRoutePhoto, dispWidth, dispHeight)
		Im = miscUtils.readScaleImage(matchRoutePhoto, dispWidth, dispHeight)

		fila = np.hstack((Iq,Im))
		displayImage = miscUtils.concatenate(fila, displayImage, 'vertical')

	cv2.namedWindow('Resultado', cv2.WINDOW_AUTOSIZE)
	cv2.imshow('Resultado', np.uint8(displayImage))
	cv2.waitKey()


