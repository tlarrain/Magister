# -*- coding: utf-8 -*-
"""
Miscelaneous Utils
Tomás Larrain A.
4 de agosto de 2014
"""
import cv2
import os
import numpy as np


def responseVector(cantPersonas, idxPerson, cantPhotosSparse):
	# Vector de representación ideal  con cantPersonas personas que tienen cantPhotosSparse fotos
	responses = np.zeros(0)
	for i in range(cantPersonas): 
		responses = np.append(responses,float(idxPerson[i])*np.ones(cantPhotosSparse))
	
	return responses


def concatenate(auxMatrix, acumMatrix, direction):
	# Concatena vertical u horizontalmente	
	if len(acumMatrix) == 0:
		acumMatrix = auxMatrix.copy()
	
	elif direction == 'vertical':
		acumMatrix = np.vstack((acumMatrix,auxMatrix))

	elif direction == 'horizontal':
		acumMatrix = np.hstack((acumMatrix,auxMatrix))	
	
	return acumMatrix


def fixedLengthString(longString, shortString):
	diff = len(longString) - len(shortString)
	outString = ' '*diff
	outString += shortString
	return outString	


def returnUnique(array):
	arrayUnique, idx = np.unique(array,return_index = True)		
	idx = np.sort(idx)
	return array[idx]	

def testResultsHeader(dataBase, useAlpha, cantPersonas, cantPhotosDict, cantIteraciones):
	# Imprime los datos generales del experimento
	header = ""
	header += "Base de Datos: " + str(dataBase) + "\n"
	header += "Se utilizó alpha: " + str(useAlpha) + "\n"
	header += "Cantidad de personas: " + str(cantPersonas) + "\n"
	header += "Fotos para diccionario: " + str(cantPhotosDict) + "\n"
	header += "Cantidad de iteraciones: " + str(cantIteraciones) + "\n\n"
	
	return header

def testResults(cantPersonas, m, m2, height, width, a, b, alpha, Q, R, L, sub, sparseThreshold, SCIThreshold, trainTimeAcumulado, testTimeAcumulado, porcAcumulado, cantIteraciones):
	# Imprime los resultados finales
	results = ""
	title = "Variables utilizadas:" + "\n"
	results += title
	results += fixedLengthString(title, "m: " + str(m)) + "\n"
	results += fixedLengthString(title, "m2: " + str(m2)) + "\n"
	results += fixedLengthString(title, "height: " + str(height)) + "\n"
	results += fixedLengthString(title, "width: " +str(width)) + "\n"
	results += fixedLengthString(title, "a: " + str(a)) + "\n"
	results += fixedLengthString(title, "b: " + str(b)) + "\n"
	results += fixedLengthString(title, "alpha: " + str(alpha)) + "\n"
	results += fixedLengthString(title, "Q: " + str(Q)) + "\n"
	results += fixedLengthString(title, "R: " + str(R)) + "\n"
	results += fixedLengthString(title, "L: " + str(L)) + "\n"
	results += fixedLengthString(title, "sub: " + str(sub)) + "\n"
	results += fixedLengthString(title, "sparseThreshold: " + str(sparseThreshold)) + "\n"
	results += fixedLengthString(title, "SCIThreshold: " + str(SCIThreshold)) + "\n"
	results += "Tiempo de entrenamiento promedio: " + str(trainTimeAcumulado/cantIteraciones) + " segundos" + "\n"
	results += "Tiempo de testing promedio: " + str(testTimeAcumulado/cantIteraciones) + " segundos/persona" + "\n"
	results += "Tiempo total del test: " + str((testTimeAcumulado*cantPersonas + trainTimeAcumulado)/60) + " minutos" + "\n"
	results += "Porcentaje acumulado: " + str(porcAcumulado/cantIteraciones) + "%\n\n\n"

	return results		

