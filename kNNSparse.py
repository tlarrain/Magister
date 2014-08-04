# -*- coding: utf-8 -*-
""""
Pruebas para el algoritmo propuesto kNN-Sparse
Tomás Larrain A.
6 de junio de 2014
"""

import numpy as np
import utils.ASRUtils as asr
import utils.miscUtils as miscUtils
import os
import time

# Parametros
m = 600			# Cantidad de patches seleccionados por foto para A
m2 = 150 		# Cantidad de patches para Matriz S
height = 100	# Alto del resize de la imagen
width = 100		# Ancho del resize de la imagen
a = 18			# Alto del patch
b = 18			# Ancho del patch
alpha = 0.5 	# Peso del centro
Q = 5			# Cluster Padres
R = 5 			# Cluser Hijos
sub = 1			# Subsample

# Inicializacion Variables
cantIteraciones = 100
porcAcumulado = 0
testTimeAcumulado = 0
trainTimeAcumulado = 0


# Datos de entrada del dataset
dataBase = "AR"
rootPath = miscUtils.getDataBasePath(dataBase)
cantFotos = 26
cantFotosA = 10
cantFotosSparse = cantFotos-cantFotosA-1

U = asr.LUT(height,width,a,b) # Look Up Table


for it in range(cantIteraciones):
# Look Up Table
	print "Iteracion ", it+1, " de ", cantIteraciones
	print "Entrenando..."
	
	beginTime = time.time()
	# Entrenamiento: Diccionario A y Parches Sparse
	YC = np.array([])
	YP = np.array([])

	# Seleccion aleatoria de individuos
	idxPerson = np.array([d for d in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, d))])
	auxIdx = np.random.permutation(len(idxPerson))[:cantPersonas]
	idxPerson = idxPerson[auxIdx]
	sujetos = len(idxPerson)
	idxFoto = np.random.permutation(cantFotos)
	
	for i in range(sujetos):
		route = os.path.join(rootPath,idxPerson[i])
		fotos = os.listdir(route)
		# fotos = np.random.permutation(fotos)
		Y = np.array([])
		for j in range(cantFotosA):
			routeFoto = os.path.join(route,fotos[idxFoto[j]])
			I = asr.readScaleImage(routeFoto,width,height)
			# Generacion de esquinas superiores izquierdas aleatorias (i,j)
			ii,jj = asr.grilla(I,m,a,b)
			Yaux = asr.patches(I,ii,jj,U,a,b,alpha,sub)
			
			

			if len(Y) == 0:
				Y = Yaux.copy()
			else:
				Y = np.vstack((Y,Yaux))
				
		Y = np.float32(Y)
		YCaux,YPaux = asr.modelling(Y,Q,R) # Clusteriza la matriz Y en padres e hijos

		if len(YC) == 0:
			YC = YCaux.copy()
			YP = YPaux.copy()
		else:
			YC = np.vstack((YC,YCaux))
			YP = np.vstack((YP,YPaux))
		

	Y = np.array([])
	Ysparse = np.array([])
	for i in range(sujetos):
		route = os.path.join(rootPath,idxPerson[i])
		fotos = os.listdir(route)
		for j in range(cantFotosSparse):
			
			idx = j+cantFotosA
			routeFoto = os.path.join(route,fotos[idxFoto[idx]])
			I = asr.readScaleImage(routeFoto,width,height)
			
			# Generacion de esquinas superiores izquierdas aleatorias (i,j)
			ii,jj = asr.grilla(I,m2,a,b)
			
			#ii = np.random.random_integers(0,height-a,size=m2)
			#jj = np.random.random_integers(0,width-b,size=m2)
			Y = asr.patches(I,ii,jj,U,a,b,alpha,sub)
			alpha1 = asr.normL1_omp(Y,YC,R)
			# alpha1 = asr.normL1_lasso(Y,YC,R)
			if len(Ysparse) == 0:
				Ysparse = alpha1.copy()
			else:
				Ysparse = np.hstack((Ysparse,alpha1))
				

	Ysparse = Ysparse.transpose()
	Ysparse = Ysparse != 0

	trainTime = time.time() - beginTime
	trainTimeAcumulado += trainTime
	aciertos = 0
	responses = np.zeros(0)
	for i in range(sujetos):
		responses = np.append(responses,float(idxPerson[i])*np.ones(cantFotosSparse))

	
	print "Testing..."
	beginTime = time.time()
	for i in range(sujetos):
		
		route = os.path.join(rootPath,idxPerson[i])
		fotos = os.listdir(route)
		routeFoto = os.path.join(route,fotos[idxFoto[cantFotos-1]])
		
		I = asr.readScaleImage(routeFoto,width,height)
		
		ii, jj = asr.grilla(I,m2,a,b)
		Y = asr.patches(I,ii,jj,U,a,b,alpha,sub)

		alpha1 = asr.normL1_omp(Y,YC,R)
		# alpha1 = asr.normL1_lasso(Y,YC,R)
		resto = float('inf')
		correcto = sujetos+1
		alphaBinary = alpha1 != 0
		alphaBinary = alphaBinary.transpose()
		for j in range(sujetos*cantFotosSparse):
			Yclass = Ysparse[j*m2:(j+1)*m2,:]
			resta = np.abs(Yclass-alphaBinary)
			restoAux = np.sum(resta)
			if restoAux < resto:
				correcto = responses[j]
				resto = restoAux

		if int(correcto) == int(idxPerson[i]):
			aciertos += 1
	
	testTime = time.time() - beginTime
	testTimeAcumulado += testTime/sujetos	
	print "Porcentaje Aciertos: " , float(aciertos)/sujetos*100,	"%\n"	
	porcAcumulado += float(aciertos)/sujetos*100


print "Experimento finalizado"
print "Tiempo de entrenamiento promedio: ", trainTimeAcumulado/cantIteraciones, " segundos"
print "Tiempo de testing promedio: ", testTimeAcumulado/cantIteraciones, " segundos/persona"
print "Porcentaje acumulado: ", porcAcumulado/cantIteraciones, "%"



