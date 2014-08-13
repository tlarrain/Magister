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


def plotGrilla(I,ii,jj,a,b,m):
	# Dibuja la grilla en colores intercalados
	for i in range(m):
		
		if i%2 == 0:
			I = miscUtils.drawPatch(I,(ii[i],jj[i]),a,b, 255, 255, 0)
		
		else:
			I = miscUtils.drawPatch(I,(ii[i],jj[i]),a,b, 0, 255, 255)

	cv2.namedWindow("Grilla")
	cv2.imshow("Grilla",np.uint8(I))
	cv2.waitKey()
	return I


width = 400
height = 400
a = 25*4
b = 25*4
m = 400

dataBase = "AR"
rootPath = miscUtils.getDataBasePath(dataBase)
routePhoto = rootPath + '01/'+'face_001_01.png'

I = cv2.imread(routePhoto)
I = cv2.resize(I,(width,height))	

iiDict,jjDict = asr.grilla_v2(height, width, a, b, m) # Grilla de m cantidad de parches

plotGrilla(I,iiDict,jjDict,a,b,m)
