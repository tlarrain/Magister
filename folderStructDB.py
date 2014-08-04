""""
Script que guarda en una estructura de carpetas las bases de datos que no los son
Tomas Larrain A.
25 de junio de 2014
Nota: muy harcodeado
"""


import shutil
import os



root = "/Users/Tomas/Developer/data/AR_old"
dst = "/Users/Tomas/Developer/data/AR"

fotoNames = os.listdir(root)

cont = 0
for i in range(100):
	if i<9:
		directory = os.path.join(dst,str(0)+str(i+1))
	else:
		directory = os.path.join(dst,str(i+1))
	if not os.path.exists(directory):
	    os.makedirs(directory)
	for j in range(26):
		src = os.path.join(root,fotoNames[cont])
		extension = os.path.splitext(src)[1]
		
		if extension == '.png':
			shutil.copyfile(src, os.path.join(directory,fotoNames[cont]))
		else:
			print extension	
		cont += 1
