#!/Users/campbell/anaconda2/bin/python

import numpy as np
from scipy import signal
from PyAstronomy import pyasl

def ffreqaxis(file):
	'''
	Extracts the frequency axis from a FITS file using the header.

	file : location of FITS file
	'''

	# extract header information
	header = getheader(file)
	CRVAL3 = header["CRVAL3"]
	CRPIX3 = header["CRPIX3"]
	CDELT3 = header["CDELT3"]
	NAXIS3 = header["NAXIS3"]

	#construct pixel array
	freqpixels = np.arange(NAXIS3)

	# transform pixels to frequency
	freqaxis =  CRVAL3 + (freqpixels-CRPIX3)*CDELT3

	return freqaxis

def freproject_2D(image1_dir,image2_dir,clean=False,order="nearest-neighbor"):
	'''
	Reprojects image1 to image2 using their FITS headers.

	Inputs:
	image1_dir : directory to image that will be reprojected
	image2_dir : directory to template image used for reprojection
	clean      : if True, creates new minimal headers based off inputs
	order      : order of interpolation (alternative options are 'bilinear', 'biquadratic', 'bicubic')

	Outputs:
	image1_data          : data to be reprojected
	image1_header        : header of image to be reprojected
	image1_data_reproj   : data of reprojected image
	image1_header_reproj : header of image1 used for reprojection (if clean=True header content is minimal)
	image2_data          : data of image used for reprojection
	image2_header_reproj : header of image2 used for reprojection (if clean=True header content is minimal)
	footprint            : a mask that defines which pixels in the reprojected image have a corresponding image in the original image
	'''

	image1_data,image1_header=fits.getdata(image1_dir,header=True)
	image2_data,image2_header=fits.getdata(image2_dir,header=True)

	if clean==True:
		image1_header_clean = fits.Header.fromkeys(["NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CRPIX1", "CRVAL1", "CDELT1", 
                                                    "CTYPE2", "CRPIX2", "CRVAL2", "CDELT2"])
		image2_header_clean = fits.Header.fromkeys(["NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CRPIX1", "CRVAL1", "CDELT1", 
                                                    "CTYPE2", "CRPIX2", "CRVAL2", "CDELT2"])

		image1_header_clean["NAXIS"]  = 2
		image1_header_clean["NAXIS1"] = image1_header['NAXIS1']
		image1_header_clean["NAXIS2"] = image1_header['NAXIS2']
		image1_header_clean["CTYPE1"] = image1_header['CTYPE1']
		image1_header_clean["CRPIX1"] = image1_header['CRPIX1']
		image1_header_clean["CRVAL1"] = image1_header['CRVAL1']
		image1_header_clean["CDELT1"] = image1_header['CDELT1']
		image1_header_clean["CTYPE2"] = image1_header['CTYPE2']
		image1_header_clean["CRPIX2"] = image1_header['CRPIX2']
		image1_header_clean["CRVAL2"] = image1_header['CRVAL2']
		image1_header_clean["CDELT2"] = image1_header['CDELT2']

		image2_header_clean["NAXIS"]  = 2
		image2_header_clean["NAXIS1"] = image2_header['NAXIS1']
		image2_header_clean["NAXIS2"] = image2_header['NAXIS2']
		image2_header_clean["CTYPE1"] = image2_header['CTYPE1']
		image2_header_clean["CRPIX1"] = image2_header['CRPIX1']
		image2_header_clean["CRVAL1"] = image2_header['CRVAL1']
		image2_header_clean["CDELT1"] = image2_header['CDELT1']
		image2_header_clean["CTYPE2"] = image2_header['CTYPE2']
		image2_header_clean["CRPIX2"] = image2_header['CRPIX2']
		image2_header_clean["CRVAL2"] = image2_header['CRVAL2']
		image2_header_clean["CDELT2"] = image2_header['CDELT2']

		image1_header_reproj = image1_header_clean
		image2_header_reproj = image2_header_clean

	else:
		image1_header_reproj = image1_header
		image2_header_reproj = image2_header

	# perform reprojection
	image1_data_reproj,footprint = reproject_interp((image1_data, image1_header_reproj), image2_header_reproj,order=order)

	return (image1_data,image1_header,image1_data_reproj,image1_header_reproj,image2_data,image2_header_reproj,footprint)

def degtosexa(ra_deg,dec_deg):
    '''
    Converts Right Ascension and Declination from decimal degrees to sexagismal format. Inputs integers, floats, lists, or arrays.
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

def sexatodeg(ra_sexa,dec_sexa):
    '''
    Converts Right Ascension and Declination from sexagismal format to decimal degrees. Inputs integers, floats,, lists, or arrays.
    '''
    
    if (isinstance(ra_sexa,str)==True):
        '''
        if input is a single coordinate.
        '''
        sexa = ra_sexa+" "+dec_sexa
        ra_deg,dec_deg = pyasl.coordsSexaToDeg(sexa)

    elif (isinstance(ra_sexa,np.ndarray)==True) or (isinstance(ra_sexa,list)==True):
        '''
        If input is an array of coordinates.
        '''
        ra_deg_list        = []
        dec_deg_list       = []
        for i in range(len(ra_sexa)):
            ra_sexa_i      = ra_sexa[i]
            dec_sexa_i     = dec_sexa[i]
            sexa_i = ra_sexa_i+" "+dec_sexa_i
            ra_deg_i,dec_deg_i = pyasl.coordsSexaToDeg(sexa_i)
            ra_deg_list.append(ra_deg_i)
            dec_deg_list.append(dec_deg_i)
        ra_deg = np.array(ra_deg_list)
        dec_deg = np.array(dec_deg_list)
     
    return ra_deg,dec_deg

def convolve(oldres_FWHM,newres_FWHM,data,header):
    '''
    Convolves data from oldres to newres.
    
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

def polgrad(Q,U):
    '''
    Constructs the spatial polarization gradient given Stokes Q and U maps.
    
    Q : Stokes Q data
    U : Stokes U data
    '''
    
    # compute Stokes spatial gradients
    Q_grad   = np.gradient(Q)
    U_grad   = np.gradient(U)
    
    # define components of spatial gradients
    Q_grad_x = Q_grad[0]
    Q_grad_y = Q_grad[1]
    U_grad_x = U_grad[0]
    U_grad_y = U_grad[1]
    
    # compute spatial polarization gradient
    polgrad  = np.sqrt(Q_grad_x**2.+Q_grad_y**2.+U_grad_x**2.+U_grad_y**2.)
    
    return polgrad

def fpolgrad_crossterms(Q,U):
    '''
    Constructs the complete spatial polarization gradient with the cross-terms included given Stokes Q and U maps.
    
    Q : Stokes Q data
    U : Stokes U data
    '''
    
    # compute Stokes spatial gradients
    Q_grad   = np.gradient(Q)
    U_grad   = np.gradient(U)
    
    # define components of spatial gradients
    Q_grad_x = Q_grad[0]
    Q_grad_y = Q_grad[1]
    U_grad_x = U_grad[0]
    U_grad_y = U_grad[1]
    
    # compute spatial polarization gradient
    a       = Q_grad_x**2.+Q_grad_y**2.+U_grad_x**2.+U_grad_y**2.
    b       = a**2. - 4.*(Q_grad_x*U_grad_y - Q_grad_y*U_grad_x)**2.
    polgrad = np.sqrt( 1.5*a + 1.5*np.sqrt(b) )
    
    return polgrad

def polgrad_rad(Q,U):
    '''
    Constructs the radial component of the spatial polarization gradient given Stokes Q and U maps.
    
    Q : Stokes Q data
    U : Stokes U data
    '''
    
    # compute Stokes spatial gradients
    Q_grad   = np.gradient(Q)
    U_grad   = np.gradient(U)
    
    # define components of spatial gradients
    Q_grad_x = Q_grad[0]
    Q_grad_y = Q_grad[1]
    U_grad_x = U_grad[0]
    U_grad_y = U_grad[1]
    
    polgrad_rad_num = (Q*Q_grad_x+U*U_grad_x)**2. + (Q*Q_grad_y+U*U_grad_y)**2.
    polgrad_rad_den = Q**2.+U**2.
    
    polgrad_rad = np.sqrt(polgrad_rad_num/polgrad_rad_den)
    
    return polgrad_rad

def polgrad_tan(Q,U):
    '''
    Constructs the tangential component of the spatial polarization gradient given Stokes Q and U maps.
    
    Q : Stokes Q data
    U : Stokes U data
    '''
    
    # compute Stokes spatial gradients
    Q_grad   = np.gradient(Q)
    U_grad   = np.gradient(U)
    
    # define components of spatial gradients
    Q_grad_x = Q_grad[0]
    Q_grad_y = Q_grad[1]
    U_grad_x = U_grad[0]
    U_grad_y = U_grad[1]
    
    polgrad_tan_num = (Q*U_grad_x-U*Q_grad_x)**2. + (Q*U_grad_y-U*Q_grad_y)**2.
    polgrad_tan_den = Q**2.+U**2.
    
    polgrad_tan = np.sqrt(polgrad_tan_num/polgrad_tan_den)
    
    return polgrad_tan

def polgrad_arg(Q,U):
    '''
    Computes the angle of the spatial polarization gradient given Stokes Q and U maps.
    
    Q : Stokes Q data
    U : Stokes U data
    '''
    
    # compute Stokes spatial gradients
    Q_grad   = np.gradient(Q)
    U_grad   = np.gradient(U)
    
    # define components of spatial gradients
    Q_grad_x = Q_grad[0]
    Q_grad_y = Q_grad[1]
    U_grad_x = U_grad[0]
    U_grad_y = U_grad[1]
    
    num = (Q_grad_x*Q_grad_y + U_grad_x*U*grad_y)*np.sqrt(Q_grad_y**2. + U_grad_y**2.)
    den = np.sqrt(Q_grad_x**2. + U_grad_x**2.)
    
    polgrad_arg = np.arctan2(num/den)
    
    return polgrad_arg

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
