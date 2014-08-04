import cv2
import numpy as np
from numpy import random
import os
import spams
import time
from scipy import signal,io
import csv


def wiener(I):
	uni = I/255.0
	W = signal.wiener(uni, mysize=(3,3))
	W = np.round(255*W)
	W = np.uint8(W)
	return W

def LUT(h,w,a,b):
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

def readScaleImage(route,width,height):
	It = cv2.imread(route)
	if It is not None:	 
		It = cv2.cvtColor(It,cv2.COLOR_BGR2GRAY)
		It = np.float32(It)
		It = cv2.resize(It,(width,height))	
		return It
	else:
		return np.zeros(0)

def patches(I,ii,jj,U,a,b,alpha,sub):
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
	ixo = np.array([ii+1,jj+1]).transpose()
	ixi = ixo.copy()
	centro = ixo +np.array([a/2,b/2]) 
	ixo = (ixo +np.array([a/2,b/2]))*alpha
	Yaux = np.hstack((Yaux,ixo))
	Y = uninorm(Yaux)
	return Y

def grilla(I,m,a,b):
	# Genera indices para extraer parches en forma de grilla (distribucion uniforme)
	h = I.shape[0]
	w = I.shape[1]
	ii = np.zeros(m)
	jj = np.zeros(m)
	divisor = int(np.floor(m ** (0.5)))
	auxIdx = 0
	for i in range(0,h-a,(h-a)/divisor):
		for j in range(0,w-b,(w-b)/divisor):
			ii[auxIdx] = i
			jj[auxIdx] = j
			auxIdx += 1
			if auxIdx >= m:
				break
		if auxIdx >= m:
			break
	return ii,jj		


def cantidadFotos(rootPath):
	# cantidad minima de fotos existentes en una clase (para que todo esten iguales)
	idx_person = os.listdir(rootPath)
	cant_fotos = float('inf')
	for d in range(len(idx_person)):
		root = os.path.join(rootPath,idx_person[d])
		files = os.listdir(root)
		if len(files)<cant_fotos:
			cant_fotos = len(files)
	return cant_fotos	

def adaptiveDictionary_v3(Y,YC,Q,R,theta):
	sujetos = YC.shape[0]/(Q*R)
	m = Y.shape[0]
	seleccion = np.zeros((sujetos,m))
	LimMat = np.zeros((R*sujetos,m))
	

	cos = np.dot(YC,Y.transpose())

	for d in range(sujetos):
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

def adaptiveDictionary_v2(patch,YC,Q,R,theta):
	# Encuentra el diccionario adaptivo basado en theta 
	sujetos = YC.shape[0]/(Q*R)
	fil_sel = np.zeros(0)
	idx_centro = []
	seleccion = []
	cos = np.dot(YC,patch)	
	for d in range(sujetos):
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


def drawPatch(I,corner,a,b):
	# Dibuja un patch de tamanio (a,b) desde la esquina corner en la imagen I.
	x = int(corner[1])
	y = int(corner[0])
	cv2.rectangle(I,(x,y),(x+a,y+b),(255,0,0))
	return I

def clasification(A,y,alpha1,R,sujetos):
	# retorna el indice de la clasificacion de un patch, y su norma L1 maxima (para calculos de SCI)
	norm1 = -1000000
	err_min = 1000000
	kmin = sujetos+1
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

def normL1_lasso(x,A,R):
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

def normL1_omp(x,A,R):
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


def seleccionar(fila,Q,R,sujetos):
	# Encuentra los sujetos seleccionados en el codigo de Matlab, basado en los centroides que se eligen
	seleccion = []
	cont = 0
	for i in range(0,Q*R,R):	
		seleccion.append(fila[i]/(Q*R))
	
	return np.array(seleccion)

def sortAndSelect(registro,tau,s,sujetos,display=False):
	seleccionFinal = np.zeros(sujetos)
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

def listaSujetos(testPath):
	ciTotal = np.zeros(0)

	with open(testPath) as f:
		data = f.readlines()

	contData = 1
	auxID_ant = 1
	while True:

		while True:

			if contData > len(data):
				break

			line = data[contData].split(' ')
			contData += 1
			ci = line[0]
			auxID = int(line[2])

			# Se comprueba que la foto sea de testing
			if auxID_ant-auxID == -1:
				auxID_ant = auxID
				break

			if auxID == 1:
				auxID_ant = auxID
				continue

			if auxID_ant-auxID == 1:
				auxID_ant = auxID

			ciTotal = np.append(ciTotal,int(ci))
			

			contData += 1

		if contData > len(data):
			break

	ciUnique, idx = np.unique(ciTotal,return_index = True)		
	idx = np.sort(idx)
	ciTotal = ciTotal[idx]
	
	return ciTotal		


def modelling(Y,Q,R):

	YP = np.zeros(0)
	YC = np.zeros(0)
	
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


def getPhotoIdx(photoName,photoList):
	for i, j in enumerate(photoList):
		if j[0] == photoName:
			return i

def readPhotoInfo(photoInfoPath):
	photoList = []
	cont = 0
	with open(photoInfoPath) as csvfile:
		photoInfo = csv.reader(csvfile, delimiter='\t', quotechar='|')
		for row in photoInfo:
			if cont < 8:
				cont += 1
				continue
			photoList.append(row[-1:])

	return photoList

def getPeopleList(peopleInfoPath):
	peopleList = []
	cont = 0
	with open(peopleInfoPath) as csvfile:
		people = csv.reader(csvfile, delimiter='\t', quotechar='|')
		for row in people:
			if cont < 8:
				cont += 1
				continue
			peopleList.append(row[1])

	peopleList = np.array(peopleList).astype(int)
	return peopleList

def getBest16List(best16Path):
	best16List = []
	cont = 0
	with open(best16Path) as csvfile:
		best16 = csv.reader(csvfile, delimiter='\t', quotechar='|')
		for row in best16:
			if cont < 4:
				cont += 1
				continue
			best16List.append(row[2:])

	best16List = np.array(best16List).astype(int)
	return best16List

def zeroFill(array,outputLength):
	length = len(array)
	rest = outputLength - length
	if rest >= 0:
		return np.append(array,np.zeros(rest))
	else:
		print "Output length smaller than original array"
		return array	

def returnUnique(array):
	arrayUnique, idx = np.unique(array,return_index = True)		
	idx = np.sort(idx)
	return array[idx]	


