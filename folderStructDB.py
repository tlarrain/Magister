# -*- coding: utf-8 -*-
""""
Script que guarda en una estructura de carpetas las bases de datos que no los son
Tomas Larrain A.
25 de junio de 2014
Nota: muy harcodeado
"""


import shutil
import os



# root = "/Users/Tomas/Dropbox/Camera Uploads"
# dst = "/Users/Tomas/Developer/data/faces/FWM"

if not os.path.exists(dst):
    os.makedirs(dst)

photoNames = os.listdir(root)

photoEx = photoNames[1]
photo = photoEx.split('_')

for i in range(1,len(photoNames)):
# 	photo = photoNames[i]
# 	actPersonID = photo.split('_')[1]
# 	directory = os.path.join(dst,actPersonID)
# 	print directory
# 	if not os.path.exists(directory): # crea la carpeta del ID de la persona si no existía anteriormente
# 		os.makedirs(directory)

# 	rootFile = os.path.join(root, photo)
# 	dstFile = os.path.join(directory, photo)
# 	shutil.copy(rootFile, dstFile)	
