import os
import numpy as np
import rht, RHT_tools
from scipy import signal
import astropy.wcs as wcs
from astropy.io import fits
from astropy import units as u
import montage_wrapper as montage
from reproject import reproject_interp
from astropy.coordinates import SkyCoord

def fdegtosexa(ra_deg,dec_deg):
	'''
	Converts Right Ascension and Declination from decimal degrees to sexagismal format. Inputs integers, floats, lists or arrays.
	'''
	
	if (isinstance(ra_deg,float)==True) or (isinstance(ra_deg,int)==True):
		'''
		if input is a single coordinate.
		'''
		sexa        = pyasl.coordsDegToSexa(ra_deg,dec_deg)
		sexa_split  = sexa.split("  ")
		ra_sexa     = sexa_split[0]
		dec_sexa    = sexa_split[1]

	elif (isinstance(ra_deg,np.ndarray)==True) or (isinstance(ra_deg,list)==True):
		'''
		If input is an array of coordinates.
		'''
		ra_sexa_list      = []
		dec_sexa_list     = []
		for i in range(len(ra_deg)):
			ra_deg_i      = ra_deg[i]
			dec_deg_i     = dec_deg[i]
			sexa_i        = pyasl.coordsDegToSexa(ra_deg_i,dec_deg_i)
			sexa_split_i  = sexa_i.split("  ")
			ra_sexa_i     = sexa_split_i[0]
			dec_sexa_i    = sexa_split_i[1]
			ra_sexa_list.append(ra_sexa_i)
			dec_sexa_list.append(dec_sexa_i)
		ra_sexa = np.array(ra_sexa_list)
		dec_sexa = np.array(dec_sexa_list)
	
	return ra_sexa,dec_sexa

def fsexatodeg(ra_sexa,dec_sexa):
	'''
	Converts Right Ascension and Declination from sexagismal format to decimal degrees.
	'''
	
	if (isinstance(ra_sexa,str)==True):
		'''
		if input is a single coordinate.
		'''
		sexa = ra_sexa+" "+dec_sexa
		ra_deg,dec_deg = pyasl.coordsSexaToDeg(sexa)

	elif (isinstance(ra_sexa,np.ndarray)==True):
		'''
		If input is an array of coordinates.
		'''
		ra_deg_list        = []
		dec_deg_list       = []
		for i in range(len(ra_sexa)):
			ra_sexa_i      = ra_sexa[i]
			dec_sexa_i     = dec_sexa[i]
			sexa_i = ra_sexa_i+" +"+dec_sexa_i
			ra_deg_i,dec_deg_i = pyasl.coordsSexaToDeg(sexa_i)
			ra_deg_list.append(ra_deg_i)
			dec_deg_list.append(dec_deg_i)
		ra_deg = np.array(ra_deg_list)
		dec_deg = np.array(dec_deg_list)
	
	return ra_deg,dec_deg

def fconvolve(oldres_FWHM,newres_FWHM,data,header):
	'''
	Convolves data from oldres to newres using standard FFT convolution.
	
	oldres : native resolution in arcminutes (FWHM)
	newres : desired resolution in arcminutes (FWHM)
	data   : data to be convolved
	header : FITS header for data
	'''
	
	# convert FWHM to standard deviations
	oldres_sigma = oldres_FWHM/(2.*np.sqrt(2.*np.log(2.)))
	newres_sigma = newres_FWHM/(2.*np.sqrt(2.*np.log(2.)))
	# construct kernel
	kernel_arcmin = np.sqrt(newres_sigma**2.-oldres_sigma**2.) # convolution theorem
	pixelsize     = header["CDELT2"]*60.                       # in arcminutes
	kernelsize    = kernel_arcmin/pixelsize                    # in pixels
	data_size_x   = data.shape[0]
	data_size_y   = data.shape[1]
	kernel_x      = signal.gaussian(data_size_x,kernelsize)
	kernel_y      = signal.gaussian(data_size_y,kernelsize)
	kernel        = np.outer(kernel_x,kernel_y)
	# normalize convolution kernel
	kernel_norm   = kernel/np.sum(kernel)
	
	# convolve data using FFT
	data_smoothed = signal.fftconvolve(data,kernel_norm,mode="same")
	
	return data_smoothed

def fmask(data,noise,snr):
	'''
	Creates a mask used to clip data based on SNR level.
	
	Inputs
	data  : data to be clipped
	noise : noise level in same units as data input
	snr   : SNR used for data clipping
	
	Outputs
	mask         : bitmask used for data clipping
	data_cleaned : masked data
	'''
	
	# calculate data SNR
	data_snr      = data/noise
	
	# create mask
	mask          = np.ones(shape=data.shape) # initialize mask
	low_snr       = np.where(data_snr<snr)    # find SNR less than input requirement
	mask[low_snr] = np.nan                    # set low SNR to nan
	
	# mask data
	data_clean    = data * mask
	
	return (mask,data_clean)

def fmaptheta_halfpolar_rad(angles):
	'''
	Maps angles from [0,2*pi) to [0,pi).

	Input
	angles : array of angles in radians to be mapped
	'''

	# map angles within [pi,2*pi) to [0,pi)
	angles_rad[(angles_rad>=1.) & (angles_rad!=2.)] -= 1.
	# map 2*pi to 0
	angles_rad[angles_rad==2.] -= 2.

	return angles_rad

def fmaptheta_halfpolar_deg(angles):
	'''
	Maps angles from [0,360) to [0,180).

	Input
	angles : array of angles in degrees to be mapped
	'''

	# map angles within [180,360) to [0,180)
	angles_deg[(angles_deg>=180.) & (angles_deg!=360.)] -= 180.
	# map 360 to 0
	angles_deg[angles_deg==360.] -= 360.

	return angles_deg

def fgradient(x):
	'''
	Constructs the spatial gradient.
	
	x : 2-dimensional input map
	'''
	
	# compute spatial gradients
	grad_xy = np.gradient(x)
	
	# define components of spatial gradient
	grad_x = grad_xy[0]
	grad_y = grad_xy[1]
	
	# compute total spatial gradient map
	grad = np.sqrt(grad_x**2. + grad_y**2.)
	
	return grad

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
    Converts AB to Vega magnitude scale.
    '''
    mag_Vega = mag_AB - zeropoint
    return mag_Vega

def fconvert_AB_vega(mag_AB,zeropoint):
	'''
	Converts AB magnitudes to the Vega magnitude scale.
	'''
	mag_Vega = mag_AB - zeropoint
	return mag_Vega
