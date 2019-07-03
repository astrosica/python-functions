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
