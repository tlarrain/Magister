# -*- coding: utf-8 -*-
"""
Image processing Utils
Tomás Larrain A.
3 de octubre de 2014
"""
import cv2
import os
import numpy as np
from scipy import signal

def readScaleImage(route, width, height, tanTriggs=False, tipo='bw'):
	# Lee la imagen de route y la escala segun width y height. El tipo 'bw' determina que sea en blanco y negro
	It = cv2.imread(route)

	if tanTriggs:
		It = tanTriggsPreprocessing(It)

	if It is not None:
		
		if tipo == 'bw' and len(It.shape) > 2:
			It = cv2.cvtColor(It,cv2.COLOR_BGR2GRAY)
		
		It = np.float32(It)
		It = cv2.resize(It,(width,height))	
		return It
	else:
		return np.zeros(0)


def tanTriggsPreprocessing(image, alpha = 0.1, tau = 10.0, gamma = 0.2, sigma0 = 1, sigma1 = 2):
	# Tan-Triggs normalizacion
	if len(image.shape) > 1:
		image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	a,b = image.shape
	I = gammaCorrection(image, (0, 1), (0, 255), gamma)
	
	# Difference of Gaussians
	I = DoG(I, sigma0, sigma1)
	
	# tanh mapping
	I = tanTriggsNormalization(I, alpha, tau)

	I = normalize8(I)
	return np.uint8(I)

def tanTriggsNormalization(I, alpha, tau):
	# Contrast EQ
	# First Stage
	tmp = np.power(abs(I), alpha)
	meanI = np.mean(tmp)
	I = I / np.power(meanI, 1.0/alpha)
	
	# Second Stage
	tmp = np.power(np.minimum(abs(I), tau), alpha)
	meanI = np.mean(tmp)
	I = I / np.power(meanI, 1.0/alpha)
	I = tau*np.tanh(I/tau)

	return I
	

def gammaCorrection(I, inInterval, outInterval, gamma):
	
	I = np.float64(I)
	a,b = I.shape
	I = adjustRange(I, inInterval)

	I = np.power(I, gamma)
	Y = adjustRange(I,outInterval)

	return Y


def adjustRange(I, interval = (0, 255)):
	
	I = np.float64(I)
	a,b = I.shape

	minNew = interval[0]
	maxNew = interval[1]

	maxOld = np.amax(I)
	minOld = np.amin(I)

	Y = ((maxNew - minNew)/(maxOld-minOld))*(I-minOld)+minNew

	return Y

def DoG(I, sigma0, sigma1):
	# Difference of Gaussians
	kernelSize0 = int(2*np.ceil(3*sigma0)+1)
	kernelSize1 = int(2*np.ceil(3*sigma1)+1)

	
	gaussian0 = cv2.GaussianBlur(I, (kernelSize0, kernelSize0), sigma0)
	gaussian1 = cv2.GaussianBlur(I, (kernelSize1, kernelSize1), sigma1)
	return gaussian0 - gaussian1


def normalize8(I):

	I = np.float64(I)
	a,b = I.shape
	maxI = np.amax(I)
	minI = np.amin(I)

	Y = np.ceil(((I - minI)/(maxI-minI))*255)

	return Y

def wiener(I):
	# Filtro de Wiener para imágenes
	uni = I/255.0
	W = signal.wiener(uni, mysize=(3,3))
	W = np.round(255*W)
	W = np.uint8(W)
	return W
