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

def fhighlatmask(lb_coords,blim):
	'''
	Creates a mased on the input limit of Galactic latitude blim.

	Inputs
	lb_coords   : Galactic coordinates (l,b) in degrees
	blim        : lower-limit on Galactic latitude where masked data satisfies |b|>=blim
	'''

	# construct coordinate grids
	lgrid,bgrid    = lb_coords.l.deg,lb_coords.b.deg

	# create mask
	mask           = np.ones(shape=bgrid.shape)
	ii             = np.abs(bgrid)<blim
	mask[ii]       = float("NaN")

	return mask

def fmask2DEQhighlat(filedir,blim):
	'''
	'''

	data      = fits.getdata(filedir)

	# create Galactic coordinate grid
	lb_coords = fcoordgrid_EQtoGAL(filedir)
	# create mask
	mask      = fhighlatmask(lb_coords,blim)
	# mask data
	data_masked = data*mask

	return data_masked

def fheader_3Dto2D(filedir_in,filedir_out,write=False):
    '''
    Transforms a 3D FITS header to a 2D FITS header by changing the appropriate keywords.

    Inputs
    filedir_in  : input file directory
    filedir_out : output file directory
    overwrite   : overwrite file boolean (default=True)
    
    '''

    data,header = fits.getdata(filedir_in,header=True)

    header_keys = header.keys()
    header["NAXIS"]=2

    keys_3D = ["NAXIS3","CDELT3","CROTA3","CRPIX3","CRVAL3","CTYPE3"]

    for key in keys_3D:
        if key in header_keys:
            del header[key]

    if write==True:
	    fits.writeto(filedir_out,data,header,overwrite=True)

    return header

def fslice3DFITS(filedir_in,dir_out,units="kms",verbose=True):
	'''
	Slices a 3D FITS data cube along the third axis and saves each 2D image as a separate FITS file.

	Inputs
	filedir_in : file directory of input FITS data cube
	dir_out    : directory where 2D image slices will be stored
	units      : units of third axis in FITS data cube
	'''

	# extract FITS data
	data,header=fits.getdata(filedir_in,header=True)

	# create velocity axis
	third_axis       = ffreqaxis(filedir_in)

	# remove 3D information from FITS header
	header_2D = fheader_3Dto2D(filedir_in,None)

	# iterate through each channel
	for i in range(data.shape[0]):
		third_axis_i = third_axis[i]*1E-3
		data_i       = data[i]
		fname        = os.path.basename(filedir_in)+"_"+str(third_axis_i)+"_"+units+".fits"
		fdir         = dir_out+fname
		if verbose==True:
			print "writing "+fdir+"..."
		fits.writeto(fdir,data_i,header_2D,overwrite=True)

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
