# -*- coding: utf-8 -*-
"""
Pruebas para el algoritmo propuesto kNN-Sparse
Tomás Larrain A.
6 de junio de 2014
"""
import utils.miscUtils as miscUtils
import utils.testUtils as testUtils
import time
import numpy as np
# Parámetros
m = 1225 				# Cantidad de patches seleccionados por foto para A
m2 = 900				# Cantidad de patches para Matriz S
height = 100			# Alto del resize de la imagen
width = 100				# Ancho del resize de la imagen
w = 20					# Alto del patch
alpha = 0.5   			# Peso del centro
Q = 50					# Cluster Padres
R = 40					# Cluser Hijos
L = 4					# Cantidad de elementos en repr. sparse
SCIThreshold = 0.1		# Umbral de seleccion de patches
tanTriggs = True		# Utilizar normalizacion de Tan-Triggs
grilla = True

# Datos de entrada del dataset
dataBase = "MPIE"

# Datos de entrada del Test
cantIteraciones = 50
cantPersonas = 68	 	
cantPhotosDict = 12

# Variables de display
displayResults = True
display = False			# Desplegar resultados
dispWidth = 30			# Ancho de las imágenes desplegadas
dispHeight = 30 		# Alto de las imágenes desplegadas


sensibilidad = False
fecha = time.asctime(time.localtime(time.time()))
########################
# EXPERIMENTOS SIMPLES #
########################



testType = 'short'


if testType == 'short':
	print 'SHORT TEST'

	header = miscUtils.testResultsHeader(dataBase, tanTriggs, grilla, cantPersonas, cantPhotosDict, cantIteraciones, fecha)
	print header
	# print "Random Corners"
	variables = miscUtils.testVariables(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold)
	print variables
	trainTimeAcumulado, testTimeAcumulado, tiemposAcumulado, itResults = testUtils.mainAlgorithm(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold, tanTriggs, dataBase, 
		cantIteraciones, cantPersonas, cantPhotosDict, display, displayResults, dispHeight, dispWidth, sensibilidad, grilla)

	timeResult = miscUtils.timeResults(tiemposAcumulado, cantIteraciones)
	
	print header
	print timeResult
	results = miscUtils.testResults(cantPersonas, trainTimeAcumulado, testTimeAcumulado, itResults)
	print results

###################################################

##########################
# EXPERIMENTOS MULTIPLES # Si se quiere editar hay que ir a testUtils.py (algunos datos de entrada se crean alla)
##########################
if testType == 'long':
	print 'LONG TEST'
	testContent = testUtils.mainLongTest(m, m2, height, width, w, alpha, Q, R, L, SCIThreshold, tanTriggs, dataBase, 
		cantIteraciones, cantPersonas, cantPhotosDict, display, displayResults, dispHeight, dispWidth, sensibilidad, grilla, fecha)
	print testContent








