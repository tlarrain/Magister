# -*- coding: utf-8 -*-
"""
Miscelaneous Utils
Tomás Larrain A.
4 de agosto de 2014
"""
import cv2
import os
import numpy as np
import time

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

def testResultsHeader(dataBase, tanTriggs, grilla, cantPersonas, cantPhotosDict, cantIteraciones, fecha, cantExperimentos=1):
	# Imprime los datos generales del experimento

	header = ""
	header += fecha + "\n"
	header += "Base de Datos: " + str(dataBase) + "\n"
	header += "Se utilizó Tan-Triggs: " + str(tanTriggs) + "\n"
	header += "Se utilizó Grilla: " + str(grilla) + "\n"
	header += "Cantidad de personas: " + str(cantPersonas) + "\n"
	header += "Fotos para diccionario: " + str(cantPhotosDict) + "\n"
	header += "Cantidad de experimentos: " + str(cantExperimentos) + "\n"
	header += "Cantidad de iteraciones por experimento: " + str(cantIteraciones) + "\n\n"
	
	return header

def testResults(cantPersonas, trainTimeAcumulado, testTimeAcumulado, itResults):
	# Imprime los resultados finales
	cantIteraciones = itResults.shape[0]
	porcAcumulado = np.mean(itResults)
	results = ""
	results += "Tiempo de entrenamiento promedio: " + str(trainTimeAcumulado/cantIteraciones) + " segundos" + "\n"
	results += "Tiempo de testing promedio: " + str(testTimeAcumulado/cantIteraciones) + " segundos/persona" + "\n"
	results += "Tiempo total del test: " + str((testTimeAcumulado*cantPersonas + trainTimeAcumulado)/60) + " minutos" + "\n"
	results += "Porcentaje acumulado: " + str(porcAcumulado) + "%\n"
	results += "Desviación estándar: " + str(np.std(itResults)) + "%\n\n\n"
	return results		


def timeResults(tiemposAcumulado, cantIteraciones):
	
	tiemposAcumulado = tiemposAcumulado/cantIteraciones
	timeResult = ""
	timeResult += "Tiempos promedio:\n "
	timeResult += "Lectura imagen: " + str(tiemposAcumulado[0]) + " segundos/persona\n"
	timeResult += "Extracción patches: " + str(tiemposAcumulado[1]) + " segundos/persona\n"
	timeResult += "Indices adaptivos: " + str(tiemposAcumulado[2]) + " segundos/persona\n"
	timeResult += "Extraccion fingerprint: " + str(tiemposAcumulado[3]) + " segundos/persona\n"
	timeResult += "Indices maximos fp: " + str(tiemposAcumulado[4]) + " segundos/persona\n"
	timeResult += "Seleccion fp: " + str(tiemposAcumulado[5]) + " segundos/persona\n"
	timeResult += "Clasificacion: " + str(tiemposAcumulado[6]) + " segundos/persona\n"
	timeResult += "Cantidad de patches seleccionados: " + str(tiemposAcumulado[7]) + "\n\n"

	return timeResult

def testVariables(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold):
	# Imprime las variables utilizadas
	results = ""
	title = "Variables utilizadas:"
	results += title  + "\n"
	results += fixedLengthString(title, "m: " + str(m)) + "\n"
	results += fixedLengthString(title, "m2: " + str(m2)) + "\n"
	results += fixedLengthString(title, "height: " + str(height)) + "\n"
	results += fixedLengthString(title, "width: " +str(width)) + "\n"
	results += fixedLengthString(title, "w: " + str(w)) + "\n"
	results += fixedLengthString(title, "alpha: " + str(alpha)) + "\n"
	results += fixedLengthString(title, "Q: " + str(Q)) + "\n"
	results += fixedLengthString(title, "R: " + str(R)) + "\n"
	results += fixedLengthString(title, "L: " + str(L)) + "\n"
	results += fixedLengthString(title, "SCIThreshold: " + str(SCIThreshold)) + "\n"

	return results

