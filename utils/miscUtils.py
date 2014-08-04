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