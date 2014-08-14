# -*- coding: utf-8 -*-
import utils.ASRUtils as asr
import numpy as np

def fixedLengthString(longString, shortString):
	diff = len(longString) - len(shortString)
	outString = ' '*diff
	outString += shortString
	return outString


height = 100
width = 100

a = 25
b = 25
m = 625
alpha = 1
sub = 1
I = np.ones((height,width))
cont = height*width


U = asr.LUT(height,width,a,b) # Look Up Table
ii,jj = asr.grilla_v2(height,width,a,b,m) # generacion de esquinas superiores izquierdas aleatorias (i,j)

Yaux = asr.patches(I, ii, jj, U, a, b, alpha, sub) # extracción de parches

