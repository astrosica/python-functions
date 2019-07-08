import numpy as np

def fmagerrtosnr(magerr):
	'''
	Converts photometric magnitude uncertainty to signal-to-noise ratio.
	'''
	magsnr = 1./magerr
	return magsnr

def fmagsnrtoerr(magsnr):
	'''
	Converts photometric magnitude signal-to-noise ratio to uncertainty.
	'''
	magerr = 1./abs(magsnr)
	return magerr

def fconvert_AB_vega(mag_AB,zeropoint):
	'''
	Converts AB magnitudes to the Vega magnitude scale.
	'''
	mag_Vega = mag_AB - zeropoint
	return mag_Vega

def fcolour_err(mag1,mag1err,mag2,mag2err):
	'''
	Computes photometric colour and its uncertainty.
	'''
	
	colour = mag1 - mag2
	colour_err = np.sqrt(mag1err**2. + mag2err**2.)
	
	return colour,colour_err