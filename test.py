# -*- coding: utf-8 -*-
"""
Test script
Tomás Larrain A.
1 de octubre de 2014
"""

import numpy as np
import utils.ASRUtils as asr
import utils.miscUtils as miscUtils
import utils.displayUtils as displayUtils
import utils.magisterUtils as magisterUtils
import utils.dataBaseUtils as dataBaseUtils
import utils.imageUtils as imageUtils
import os
import time
import cv2
from sklearn.neighbors import NearestNeighbors



dataBase = "Yale"

dataBasePath, cantPhotosPerPerson = dataBaseUtils.getDataBasePath(dataBase)

idxPerson = "027"
idxPhoto = np.array([0,2,3,6,13,22])

route = os.path.join(dataBasePath, idxPerson)
width = 168
height = 192
photos = os.listdir(route)
for i in range(len(idxPhoto)):
	routePhoto = os.path.join(route, photos[idxPhoto[i]])
	I = imageUtils.readScaleImage(routePhoto, width, height, tanTriggs=True) 
	cv2.imwrite(photos[idxPhoto[i]] + "_TT", I)
