# -*- coding: utf-8 -*-
"""
Librería para algoritmo ASR y derivados
Tomás Larrain A.
"""

import cv2
import numpy as np
import spams
import miscUtils
import imageUtils
from scipy.spatial import distance as dist
from sklearn.neighbors import KNeighborsClassifier
from scipy import io
import os

def LUT(height, width, w):
	# Genera una Look Up Table para extraer parches de wxw de una imagen de heightxwidth
	U = np.zeros(((height-w+1)*(width-w+1),w*w))
	I =np.zeros((height,width))
	count = 0
	for i in range(height):
		for j in range(width):
			I[i,j] = count
			count+=1
	count = 0		
	for i in range(height-w+1):
		for j in range(width-w+1):
			fila = I[i:i+w,j:j+w].transpose().flatten()
			U[count,:] = fila
			count+=1	
	return U			


def uninorm(Y):
	# Normaliza las filas de Y
	for i in range(Y.shape[0]):
		Y[i,:] = cv2.normalize(Y[i,:]).flatten()	
	return Y


def randomCorners(height, width, w, m):
	# esquinas aleatorias	
	ii = np.random.random_integers(0,height-w,size=m)
	jj = np.random.random_integers(0,width-w,size=m)

	return ii, jj


def patches(I, ii, jj, U, w, alpha, sub,useAlpha=True):
	# Extrae todos los patches cuya esquina superior izquierda es (ii,jj)
	ii = ii.astype(int)
	jj = jj.astype(int)
	h = I.shape[1]-w+1
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
		centro = ixo +np.array([w/2,w/2]) 
		ixo = (ixo +np.array([w/2,w/2]))*alpha
		Yaux = np.hstack((Yaux,ixo))

	Y = uninorm(Yaux)
	return Y


def grilla(height, width, w, m):
	# Genera indices para extraer parches en forma de grilla (distribucion uniforme)

	ii = np.zeros(m)
	jj = np.zeros(m)
	divisor = int(np.floor(m ** (0.5)))
	auxIdx = 0
	pasoI = max(1,(height-w)/divisor)
	pasoJ = max(1,(width-w)/divisor)
	print pasoI, pasoJ
	for i in range(0,height-w+1,pasoI):
		for j in range(0,width-w+1,pasoJ):
			ii[auxIdx] = i
			jj[auxIdx] = j
			auxIdx += 1
			if auxIdx >= m:
				break
		if auxIdx >= m:
			break
	return ii,jj		

def grilla_v2(height, width, w, m):
	# Pensada para usar m como cuadrado perfecto (tiene sentido si las fotos que se usan son cuadradas)

	ii = np.zeros(m)
	jj = np.zeros(m)
	cantidadPorEje = int(np.floor(m ** (0.5)))
	
	ejeI = np.floor(np.linspace(0,height-w,cantidadPorEje))
	ejeJ = np.floor(np.linspace(0,width-w,cantidadPorEje))
	auxIdx = 0
	
	for i in ejeI:
		for j in ejeJ:
			ii[auxIdx] = i
			jj[auxIdx] = j
			auxIdx += 1
			
	return ii[:cantidadPorEje**2],jj[:cantidadPorEje**2]	


def generateDictionary(dataBasePath, idxPerson, idxPhoto, iiDict, jjDict, Q, R, U, width, height, w, alpha, sub, useAlpha, cantPhotosDict, tanTriggs):
	# Genera diccionario con parches de fotos
	cantPersonas = len(idxPerson)
	YC = np.array([])
	YP = np.array([])
	for i in range(cantPersonas):

		# Ruta de la persona i y lista de todas sus fotos
		route = os.path.join(dataBasePath, idxPerson[i])
		photos = os.listdir(route)
		
		Y = np.array([])
		
		for j in range(cantPhotosDict):
			
			routePhoto = os.path.join(route, photos[idxPhoto[i,j]]) # ruta de la foto j
			I = imageUtils.readScaleImage(routePhoto, width, height, tanTriggs=tanTriggs) # lectura de la imagen
					
			Yaux = patches(I, iiDict, jjDict, U, w, alpha, sub, useAlpha) # extracción de parches
		
			# Concatenación de matrices Yaux
			Y = miscUtils.concatenate(Yaux, Y, 'vertical')

		
		YCaux,YPaux = modelling(Y, Q, R) # Clusteriza la matriz Y en padres e hijos
			
		# Concatenación de matrices YC e YP
		YC = miscUtils.concatenate(YCaux, YC, 'vertical')
		YP = miscUtils.concatenate(YPaux, YP, 'vertical')

	YC = uninorm(YC)
	
	return YC


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
		
		
	LimMat = LimMat.astype(int).transpose()		
	seleccion = np.array(seleccion).transpose()
	return LimMat, seleccion


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
		
		
	LimMat = LimMat.astype(int).transpose()		
	seleccion = np.array(seleccion).transpose()
	return LimMat, seleccion


def adaptiveDictionary_v4(Y, YC, Q, R):
	# Version para magister
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
		
		for k in range(m):
			LimMat[d*R:(d+1)*R,k] = np.array(range(lim_inf[k],lim_sup[k]))
		
		
	LimMat = LimMat.astype(int).transpose()		
	
	return LimMat	

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
	# minimizacion L1-Lasso
	X = np.asfortranarray(np.matrix(x).transpose())
	D = np.asfortranarray(A.transpose())
	X = np.float64(X)
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
	# minimizacion L1-OMP
	# x = np.float64(x)
	# A = np.float64(A)

	X = np.asfortranarray(np.matrix(x).transpose(), dtype = 'f8')
	D = np.asfortranarray(np.matrix(A).transpose(), dtype = 'f8')
	
	numThreads = -1
	eps = 0.0
	alpha = spams.omp(X, D = D, L = R, eps = eps, return_reg_path = False, numThreads = numThreads)
	#alpha = spams.omp(X, D, L = R)
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
	# Construye el diccionario
	YP = np.zeros(0)
	YC = np.zeros(0)
	Y = np.float32(Y)
	
	criteria = (cv2.TERM_CRITERIA_MAX_ITER,1000, 1e-3)
	# criteria = (cv2.TERM_CRITERIA_EPS,1000, 0)
	ret,labels,centers = cv2.kmeans(Y,Q,criteria,1,cv2.KMEANS_PP_CENTERS)
	YP = centers.copy()
	
	# Clusteriza en los Hijos
	for i in range(Q):
		labels = labels.flatten()
		idx = np.nonzero(labels==i)[0]
		z = Y[idx,:].copy()
		if R<len(idx): # Si el cluster es muy chico, lo rellena con su centro de masa
			ret2,labels2,centers2 = cv2.kmeans(z,R,criteria,1,cv2.KMEANS_PP_CENTERS)
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


def SCI(alpha, Q, R, cantSujetos, L):
	# coeficiente SCI
	maxSum = 0

	for i in range(cantSujetos): 
		alphaClass = alpha[i*Q*R:(i+1)*Q*R]
		auxSum = cv2.norm(alphaClass,cv2.NORM_L1)
		
		if auxSum > maxSum:
			maxSum = auxSum
	
	     
	SCI = (float(maxSum)/L*cantSujetos - 1)/(cantSujetos-1)
	
	
	return SCI



