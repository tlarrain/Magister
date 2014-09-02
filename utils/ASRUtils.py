# -*- coding: utf-8 -*-
"""
Librería para algoritmo ASR y derivados
Tomás Larrain A.
"""

import cv2
import numpy as np
import spams
import miscUtils
from scipy import signal
from scipy.spatial import distance as dist
from sklearn.neighbors import KNeighborsClassifier
import os

def LUT(h, w, a, b):
	# Genera una Look Up Table para extraer parches de axb de una imagen de hxw
	U = np.zeros(((h-a+1)*(w-b+1),a*b))
	I =np.zeros((h,w))
	count = 0
	for i in range(h):
		for j in range(w):
			I[i,j] = count
			count+=1
	count = 0		
	for i in range(h-a+1):
		for j in range(w-b+1):
			fila = I[i:i+a,j:j+b].transpose().flatten()
			U[count,:] = fila
			count+=1	
	return U			


def uninorm(Y):
	# Normaliza las filas de Y
	for i in range(Y.shape[0]):
		Y[i,:] = cv2.normalize(Y[i,:]).flatten()	
	return Y


def randomCorners(h, w, a, b, m):
	# esquinas aleatorias	
	ii = np.random.random_integers(0,h-a,size=m)
	jj = np.random.random_integers(0,w-b,size=m)

	return ii, jj


def patches(I, ii, jj, U, a, b, alpha, sub,useAlpha=True):
	# Extrae todos los patches cuya esquina superior izquierda es (ii,jj)
	ii = ii.astype(int)
	jj = jj.astype(int)
	h = I.shape[1]-b+1
	kk = ii*h+jj
	Iv = np.reshape(I,I.shape[0]*I.shape[1])
	n = len(kk)
	m1 = U.shape[1]
	Yaux = np.zeros((n,m1))
	Yaux = Iv[U[kk,:].astype(int)]
	Yaux = Yaux[:,0::sub]

	if useAlpha:	
		ixo = np.array([ii+1,jj+1]).transpose()
		ixi = ixo.copy()
		centro = ixo +np.array([a/2,b/2]) 
		ixo = (ixo +np.array([a/2,b/2]))*alpha
		Yaux = np.hstack((Yaux,ixo))

	Y = uninorm(Yaux)
	return Y


def grilla(h, w, a, b, m):
	# Genera indices para extraer parches en forma de grilla (distribucion uniforme)

	ii = np.zeros(m)
	jj = np.zeros(m)
	divisor = int(np.floor(m ** (0.5)))
	auxIdx = 0
	pasoI = max(1,(h-a)/divisor)
	pasoJ = max(1,(w-b)/divisor)
	print pasoI, pasoJ
	for i in range(0,h-a+1,pasoI):
		for j in range(0,w-b+1,pasoJ):
			ii[auxIdx] = i
			jj[auxIdx] = j
			auxIdx += 1
			if auxIdx >= m:
				break
		if auxIdx >= m:
			break
	return ii,jj		

def grilla_v2(h, w, a, b, m):
	# Pensada para usar m como cuadrado perfecto (tiene sentido si las fotos que se usan son cuadradas)

	ii = np.zeros(m)
	jj = np.zeros(m)
	cantidadPorEje = int(np.floor(m ** (0.5)))
	
	ejeI = np.floor(np.linspace(0,h-a,cantidadPorEje))
	ejeJ = np.floor(np.linspace(0,w-b,cantidadPorEje))
	auxIdx = 0
	
	for i in ejeI:
		for j in ejeJ:
			ii[auxIdx] = i
			jj[auxIdx] = j
			auxIdx += 1
			
	return ii,jj	


def adaptiveDictionary_v3(Y, YC, Q, R, theta):
	cantPersonas = YC.shape[0]/(Q*R)
	m = Y.shape[0]
	seleccion = np.zeros((cantPersonas,m))
	LimMat = np.zeros((R*cantPersonas,m))
	

	cos = np.dot(YC,Y.transpose())

	for d in range(cantPersonas):
		cos1 = cos[d*Q*R:(d+1)*Q*R,:]
		minimo = 1-np.amax(cos1,axis=0)

		lim_inf = d*Q*R+(np.argmax(cos1,axis=0)/R)*R
		lim_sup = lim_inf + R
		
		sujSel = minimo < theta
		minSel = np.where(minimo < theta)[0]
		
		for k in range(m):
			if sujSel[k]:
				LimMat[d*R:(d+1)*R,k] = np.array(range(lim_inf[k],lim_sup[k]))
		
		seleccion[d,:] = sujSel
		
		
	LimMat = LimMat.astype(int)		
	seleccion = np.array(seleccion)
	return LimMat, seleccion


def adaptiveDictionary_v2(patch, YC, Q, R, theta):
	# Encuentra el diccionario adaptivo basado en theta 
	cantPersonas = YC.shape[0]/(Q*R)
	fil_sel = np.zeros(0)
	idx_centro = []
	seleccion = []
	cos = np.dot(YC,patch)	
	for d in range(cantPersonas):
		cos1 = cos[d*Q*R:(d+1)*Q*R]
		minimo = 1-max(cos1)

		lim_inf = d*Q*R+(np.argmax(cos1)/R)*R
		lim_sup = lim_inf + R
		if minimo<theta:
			seleccion.append(d)
			fil_sel = np.concatenate((fil_sel,range(lim_inf,lim_sup)))
			idx_centro.append(lim_sup/R)
	fil_sel = fil_sel.astype(int)		
	A = YC[fil_sel,:]
	idx_centro = np.array(idx_centro)
	seleccion = np.array(seleccion)			
	return A,seleccion, fil_sel			



def clasification(A, y, alpha1, R, cantPersonas):
	# retorna el indice de la clasificacion de un patch, y su norma L1 maxima (para calculos de SCI)
	norm1 = -1000000
	err_min = 1000000
	kmin = cantPersonas+1
	for k in range(len(alpha1)/R):
		alpha1 = alpha1.flatten()
		delta = alpha1[R*k:R*(k+1)]
		norm_delta = cv2.norm(delta,cv2.NORM_L1)
		if norm1<norm_delta:
			norm1 = cv2.norm(delta,cv2.NORM_L1)
		A1 = A[R*k:R*(k+1),:].transpose()
		reconst = np.dot(A1,delta)
		error = cv2.norm(y-reconst)
		if error < err_min:
			err_min = error
			kmin = k
	return kmin, norm1

def normL1_lasso(x, A, R):
	# minimizacion L1
	X = np.asfortranarray(np.matrix(x).transpose())
	D = np.asfortranarray(A.transpose())
	X = np.float32(X)
	numThreads = -1
	eps = 0.0
	param = {
    'lambda1' : 0.15, 
    'numThreads' : -1, 
    'mode' : 2 ,
    'L': R}        
	(alpha,path) = spams.lasso(X,D = D,return_reg_path = True,**param)

	alpha = np.array(alpha.todense())
	return alpha

def normL1_omp(x, A, R):
	# minimizacion L1
	X = np.asfortranarray(np.matrix(x).transpose())
	D = np.asfortranarray(A.transpose())
	X = np.float64(X)
	D = np.float64(D)
	numThreads = -1
	eps = 0.0
	alpha = spams.omp(X,D = D,L = R,eps = eps,return_reg_path = False,numThreads = numThreads)

	alpha = np.array(alpha.todense())
	return alpha


def sortAndSelect(registro, tau, s, cantPersonas, display=False):
	seleccionFinal = np.zeros(cantPersonas)
	idx_sort = np.argsort(registro[1,:])
	sort = registro[:,idx_sort]
	idx = np.nonzero(sort[1,:]>=tau)[0]
	sort = sort[:,idx]
	sort_final = sort[:,-s:]
	
	for i in sort_final[0,:]:
		seleccionFinal[i]+=1
	if display:
		print sort_final	
	return seleccionFinal
	

def modelling(Y, Q, R):

	YP = np.zeros(0)
	YC = np.zeros(0)
	Y = np.float32(Y)
	
	criteria = (cv2.TERM_CRITERIA_MAX_ITER,1000, 1e-3)
	ret,labels,centers = cv2.kmeans(Y,Q,criteria,1,cv2.KMEANS_RANDOM_CENTERS)
	YP = centers.copy()
	
	# Clusteriza en los Hijos
	for i in range(Q):
		labels = labels.flatten()
		idx = np.nonzero(labels==i)[0]
		z = Y[idx,:].copy()
		if R<len(idx): # Si el cluster es muy chico, lo rellena con su centro de masa
			ret2,labels2,centers2 = cv2.kmeans(z,R,criteria,1,cv2.KMEANS_RANDOM_CENTERS)
			CC = uninorm(centers2)
		else:
			CC = z.copy()

		Rc = np.tile(CC,(R,1))
		Rc = Rc[0:R,:]
		if len(YC) == 0:
			YC = Rc.copy()
		else:
			YC = np.vstack((YC,Rc))
	
	return YC,YP



def fingerprint(I, U, YC, ii, jj, R, a, b, alpha, sub, useAlpha=True, tipo='omp'):
	# Generación de fingerprint sparse de la imagen I	
	height = I.shape[0]
	width = I.shape[1]
	Y = patches(I, ii, jj, U, a, b, alpha, sub, useAlpha)
	if tipo == 'omp':
		alpha1 = normL1_omp(Y, YC, R)
		alpha1 = np.reshape(alpha1,(alpha1.shape[0]*alpha1.shape[1],1))
		return alpha1

	if tipo == 'lasso':
		alpha1 = normL1_lasso(Y, YC, R)
		alpha1 = np.reshape(alpha1,(alpha1.shape[0]*alpha1.shape[1],1))
		return alpha1

	else:
		print "Tipo de minimización no válido"
		exit()	


def distance(array1, array2, tipo):
	# Distintos tipos de métricas de distancia
	eps = 1e-5
	length = len(array1)
	
	if tipo == 'euclidean':
		return	dist.euclidean(array1,array2)

	if tipo == 'hamming':
		return dist.hamming(array1,array2)*length

	if tipo == 'absDiff':
		return np.sum(np.abs(array1-array2),axis=1)
	
	if tipo == 'chiSquare':
		return np.sum(0.5*np.divide(((array1-array2)**2),(array1+array2+eps)))
	else:
		print "Tipo de distancia no válido"
		exit()


def clasifier(Ysparse, alpha1, responses, distType):
	# Clasificador original del algoritmo
	total = Ysparse.shape[0]
	resto = float('inf')
	for j in range(total):
			
		Yclass = Ysparse[j, :] # matriz sparse que representa la foto
		
		restoAux = distance(Yclass,alpha1,distType) # valor absoluto de la resta
		
		
		# Encuentra la resta con menor error
		if restoAux < resto:
			correctPhoto = j
			correctID = responses[j]
			resto = restoAux

	return correctPhoto, correctID


def testing(dataBasePath, idxPerson, idxPhoto, width, height, U, YC, Ysparse, ii, jj, R, a, b, alpha, sub, sparseThreshold, useAlpha, distType, responses):
	# testing algoritmo
	cantPersonas = idxPhoto.shape[0]
	idxTestPhoto = idxPhoto.shape[1]-1
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior
	aciertos = 0

	for i in range(cantPersonas):
	# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, R, a, b, alpha, sub, useAlpha)
		
		# Inicialización variables de testing
		resto = float('inf')
		corrPhoto = cantPersonas+1
		
		# Binarización representaciones sparse
		alpha1 = alpha1.transpose()
		if  distType != 'euclidean' and distType != 'chiSquare':
			alpha1 = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral
		# alphaBinary = alpha1 != 0 # distintas de cero
		
		
		corrPhoto, corrID = clasifier(Ysparse, alpha1, responses, distType)
		correctPhoto[i,0] = corrPhoto

		
		# Compara con vector de clasificación ideal
		if int(corrID) == int(idxPerson[i]):
			aciertos += 1
			correctPhoto[i,1] = 1

	return aciertos, correctPhoto

def testingOld(dataBasePath, cantPersonas, idxPerson, idxPhoto, width, height, U, YC, Ysparse, ii, jj, R, a, b, alpha, sub, sparseThreshold, useAlpha, distType, responses):
	# testing algoritmo
	
	idxTestPhoto = len(idxPhoto)-1
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior
	aciertos = 0

	for i in range(cantPersonas):
	# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, R, a, b, alpha, sub, useAlpha)
		
		# Inicialización variables de testing
		resto = float('inf')
		corrPhoto = cantPersonas+1
		
		# Binarización representaciones sparse
		alpha1 = alpha1.transpose()
		if  distType != 'euclidean' and distType != 'chiSquare':
			alpha1 = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral
		# alphaBinary = alpha1 != 0 # distintas de cero
		
		
		corrPhoto, corrID = clasifier(Ysparse, alpha1, responses, distType)
		correctPhoto[i,0] = corrPhoto

		
		# Compara con vector de clasificación ideal
		if int(corrID) == int(idxPerson[i]):
			aciertos += 1
			correctPhoto[i,1] = 1

	return aciertos, correctPhoto

def testing_NBNN(dataBasePath, idxPerson, idxPhoto, n_neighbors, width, height, U, YC, Ysparse, ii, jj, Q, R, a, b, alpha, sub, sparseThreshold, useAlpha, distType, responses):
	# testing algoritmo
	cantPersonas = idxPhoto.shape[0]
	idxTestPhoto = idxPhoto.shape[1]-1
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior
	aciertos = 0

	for i in range(cantPersonas):
	# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, R, a, b, alpha, sub, useAlpha)
		
		# Inicialización variables de testing
		resto = float('inf')
		corrPhoto = cantPersonas+1
		
		# Binarización representaciones sparse
		alpha1 = alpha1.transpose()
		if  distType != 'euclidean' and distType != 'chiSquare':
			alpha1 = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral
		# alphaBinary = alpha1 != 0 # distintas de cero
		
		
		corrID = NBNN(Ysparse, alpha1, responses, Q, R, distType, n_neighbors)
		
		# Compara con vector de clasificación ideal
		if int(corrID) == int(idxPerson[i]):
			aciertos += 1
			
	return aciertos

def testing_Scikit(dataBasePath, idxPerson, idxPhoto, n_neighbors, width, height, U, YC, Ysparse, ii, jj, R, a, b, alpha, sub, sparseThreshold, useAlpha, distType, responses):
	cantPersonas = idxPhoto.shape[0]
	idxTestPhoto = idxPhoto.shape[1]-1
	correctPhoto = np.zeros((cantPersonas,2)) # para propositos del despliegue de las imagenes posterior
	aciertos = np.zeros(len(n_neighbors))

	Ytest = np.array([])

	for i in range(cantPersonas):
		# Ruta de la foto de testing
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		routePhoto = os.path.join(route, photos[idxPhoto[i,idxTestPhoto]])
		
		I = miscUtils.readScaleImageBW(routePhoto, width, height) # lectura de la imagne
		alpha1 = fingerprint(I, U, YC, ii, jj, R, a, b, alpha, sub, useAlpha)
		
		alpha1 = alpha1.transpose()
		# Binarización representaciones sparse
		if  distType != 'euclidean' and distType != 'chiSquare':
			alpha1 = (alpha1 < -sparseThreshold) | (alpha1 > sparseThreshold) # por umbral


		Ytest = miscUtils.concatenate(alpha1, Ytest, 'vertical')
	
	
	for n in range(len(n_neighbors)):
		neigh = KNeighborsClassifier(n_neighbors=n_neighbors[n],metric=distType) # inicializacion clasificador
		neigh.fit(Ysparse, responses) # entrenador del algoritmo
		resultIDs = neigh.predict(Ytest) # testing
		resultPhotos = neigh.kneighbors(Ytest, n_neighbors=n_neighbors[n], return_distance=False) # foto más cercana
			
	
		for i in range(cantPersonas):

			correctPhoto[i,0] = resultPhotos[i][0]	
			
			# Compara con vector de clasificación ideal
			if int(resultIDs[i]) == int(idxPerson[i]):
				aciertos[n] += 1
				correctPhoto[i,1] = 1

	return aciertos, correctPhoto	


def NBNN(Ysparse, alpha1, responses, Q, R, distType, n_neighbors):
	# Naive Bayes Nearest Neighbours
	idxPerson = returnUnique(responses)
	cantSujetos = len(idxPerson)
	subSize = Q*R*cantSujetos
	votes = np.zeros(cantSujetos)
	m = alpha1.shape[1]/subSize
	for i in range(m):
		alphaSub = alpha1[:,i*subSize:(i+1)*subSize]
		YsparseSub = Ysparse[:,i*subSize:(i+1)*subSize]
		neigh = KNeighborsClassifier(n_neighbors= n_neighbors[0],metric=distType) # inicializacion clasificador
		neigh.fit(YsparseSub, responses) # entrenador del algoritmo
		resultID = neigh.predict(alphaSub) # testing
		idx = np.where(idxPerson == resultID[0])[0]
		votes[idx] += 1

	winner = idxPerson[np.argmax(votes)]

	return winner


def returnUnique(array):
	arrayUnique, idx = np.unique(array,return_index = True)		
	idx = np.sort(idx)
	return array[idx]	


