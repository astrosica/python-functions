import numpy as np
from scipy import interpolate
from scipy import signal, spatial
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft

from matplotlib import rc
rc("text", usetex=True)

def fconvolve(oldres,newres,data,header,method="scipy"):
	'''
	Convolves an image using FFT convolution.
	Note: newres is not the size of the convolution kernel, this will be solved for.
	
	Input
	oldres : input resolution in arcminutes (FWHM)
	newres : desired resolution in arcminutes (FWHM)
	data   : image to be convolved
	header : FITS header of image
	method : method of interpolation; can be scipy or astropy (default=scipy)
	
	Output
	data_smoothed : smoothed image
	'''
	
	# convert FWHM to standard deviations
	oldres_sigma  = oldres/(2.*np.sqrt(2.*np.log(2.)))
	newres_sigma  = newres/(2.*np.sqrt(2.*np.log(2.)))
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

	# convolve
	if method=="scipy":
		data[np.isnan(data)] = 0.0
		data_smoothed = signal.fftconvolve(data,kernel_norm,mode="same")
	elif method=="astropy":
		data_smoothed = convolve_fft(data,kernel_norm,boundary="fill",fill_value=0.0,nan_treatment="interpolate",normalize_kernel=True,allow_huge=True)
	
	return data_smoothed

def fmask_basketweaving(image):
	'''
	Creates a mask for Arecibo's basketweaving artefacts in Fourier space.

	Input
	image : two-dimensional data to be masked (FFT of sensitivity map)
	x0_1  : starting x-coordinate of first (lower) line
	y0_1  : starting y-coordinate of first (lower) line
	x1_1  : ending x-coordinate of first (lower) line
	y1_1  : ending y-coordinate of first (lower) line
	x0_2  : starting x-coordinate of second (upper) line
	y0_2  : starting y-coordinate of second (upper) line
	x1_2  : ending x-coordinate of second (upper) line
	y1_2  : ending y-coordinate of second (upper) line
	num   : number of pixels in each line
	
	Output
	mask         : resulting mask
	image_masked : masked image
	'''

	NAXIS2,NAXIS1 = image.shape
	xpix,ypix     = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid   = np.meshgrid(xpix,ypix)

	xypoints = np.array([
		[0,-96,NAXIS1,83,0,-82,NAXIS1,97],
		[0,83,NAXIS1,262,0,102,NAXIS1,281],
		[0,262,NAXIS1,442,0,280,NAXIS1,460],
		[0,441,NAXIS1,620,0,455,NAXIS1,634],
		[0,620,NAXIS1,799,0,634,NAXIS1,813],
		[0,799,NAXIS1,978,0,813,NAXIS1,992],
		[0,978,NAXIS1,1157,0,992,NAXIS1,1171],
		#
		[0,83,NAXIS1,-96,0,97,NAXIS1,-82],
		[0,262,NAXIS1,83,0,281,NAXIS1,102],
		[0,441,NAXIS1,262,0,455,NAXIS1,276],
		[0,620,NAXIS1,441,0,634,NAXIS1,455],
		[0,799,NAXIS1,620,0,813,NAXIS1,634],
		[0,978,NAXIS1,799,0,992,NAXIS1,813],
		[0,1157,NAXIS1,978,0,1171,NAXIS1,992]
		])

	# initialize mask
	mask = np.full((NAXIS2,NAXIS1), True, dtype=bool)

	# iterate through each set of lines to iteratively update mask
	for i in xypoints:
		x0_1,y0_1,x1_1,y1_1,x0_2,y0_2,x1_2,y1_2 = i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7]
		# compute pixel coordinates of each line
		xpix_1, ypix_1  = np.linspace(x0_1,x1_1,NAXIS1), np.linspace(y0_1,y1_1,NAXIS1)
		xpix_2, ypix_2  = np.linspace(x0_2,x1_2,NAXIS1), np.linspace(y0_2,y1_2,NAXIS1)
		# compute mask between lines
		mask_i          = (ygrid>ypix_1) & (ygrid<ypix_2)
		mask_i          = np.invert(mask_i)
		mask           *= mask_i

	# adjust mask for non-artefact features that should not be removed
	mask[:,2684:2687]   = True # vertical line in the image center
	mask_ellipse,_      = fmask_ellipse(image,NAXIS1*0.5,NAXIS2*0.5,100,420.,6.)
	mask_ellipse        = np.invert(mask_ellipse)
	mask[mask_ellipse]  = True

	# convert mask to ones and zeros
	mask = mask.astype(float)

	# mask image
	image_masked    = image*mask

	return mask,image_masked

def fmask_snr(data,noise,snr,fill_value=np.nan):
	'''
	Computes a mask to clip data based on S/N level.
	
	Input
	data       : data to be clipped
	noise      : noise level in the same units as data input
	snr        : SNR used for data clipping
	fill_value : value to fill masked regions with
	
	Output
	mask         : bitmask used for data clipping
	data_cleaned : masked data
	'''

	# calculate data SNR
	data_snr      = data/noise

	# create mask
	mask          = np.ones(shape=data.shape) # initialize mask
	low_snr       = np.where(data_snr<snr)    # find SNR less than input requirement
	mask[low_snr] = fill_value                # set low SNR to nan
	
	# mask data
	data_clean    = data*mask

	return (mask,data_clean)

def fmask_signal(data,signal,fill_value=np.nan):
	'''
	Creates a mask used to clip data based on signal level.
	
	Input
	data       : data to be clipped
	signal     : signal used for data clipping
	fill_value : value to fill masked regions with
	
	Output
	mask         : bitmask used for data clipping
	data_cleaned : masked data
	'''

	# create mask
	mask             = np.ones(shape=data.shape) # initialize mask
	low_signal       = np.where(data<signal)     # find signal less than input requirement
	mask[low_signal] = fill_value                # set low signal to nan
	
	# mask data
	data_clean       = data*mask

	return (mask,data_clean)

def fmaptheta_halfpolar(angles,deg=False):
	'''
	Maps angles from [0,2*pi) to [0,pi) or from [0,360) to [0,180).

	Input
	angles : array of angles to be mapped
	deg    : boolean which specifies units of input angles (default unit is radian)

	Output
	angles : angles on [0,pi) or [0,180)
	'''

	if deg==False:
		# map angles within [pi,2*pi) to [0,pi)
		angles[(angles>=1.) & (angles!=2.)] -= 1.
		# map 2*pi to 0
		angles[angles==2.] -= 2.
	elif deg==True:
		# map angles within [180,360) to [0,180)
		angles[(angles>=180.) & (angles!=360.)] -= 180.
		# map 360 to 0
		angles[angles==360.] -= 360.

	return angles

def fgradient(image):
	'''
	Computes the spatial gradient of a two-dimensional image.
	
	Input
	image : a two-dimensional image

	Output
	grad : the two-dimensional spatial gradient with the same size as the input image
	'''
	
	# compute spatial gradients
	grad_y,grad_x = np.gradient(image)
	
	# compute total spatial gradient map
	grad = np.sqrt(grad_x**2. + grad_y**2.)
	
	return grad

def fmaskinterp(image,mask):
	'''
	Masks and interpolates a two-dimensional image.

	Inputs
	image : 2D array
	mask  : 2D array of the same size as image whose masked values for invalid pixels are NaNs

	Output
	image_interp : the masked and interpolated image
	'''

	# create pixel grid
	x      = np.arange(0, image.shape[1])
	y      = np.arange(0, image.shape[0])
	xx, yy = np.meshgrid(x,y)

	# create boolean mask for invalid numbers
	mask_invalid = np.isnan(mask)

	#get only the valid values
	x1        = xx[~mask_invalid]
	y1        = yy[~mask_invalid]
	image_new = image[~mask_invalid]

	# interpolate 
	image_interp = interpolate.griddata((x1, y1), image_new.ravel(),(xx, yy),method="cubic")

	return image_interp

def fFFT(data,pos=False):
	'''
	Performs a one-dimensional fast Fourier transform (FFT).

	Input
	data : input one dimensional data to transform; can be a list or an array
	pos  : if True, only return the positive-valued frequency components

	Output
	freq     : sampled frequencies
	data_fft : one-dimensional FFT
	'''

	if isinstance(data,list)==True:
		# if input data is a list, convert to array
		data = np.array(data)

	N  = len(data)
	dt = 1.

	freq     = fftpack.fftfreq(N,dt)
	data_fft = fftpack.fft(data.astype(float))

	if pos==True:
		freq     = freq[0:N/2]
		data_fft = data_fft[0:N/2]

	return freq,data_fft

def fIFFT(data,axis=-1):
	'''
	Performs a one-dimensional inverse fast Fourier transform (IFFT).

	Input
	data : input one dimensional data to transform; can be a list or an array

	Output
	data_ifft : one-dimensional FFT
	'''

	if isinstance(data,list)==True:
		# if input data is a list, convert to array
		data = np.array(data)

	data_ifft = fftpack.ifft(data, axis=axis)

	return data_ifft

def fFFT2D(data,shift=True):
	'''
	Performs a two-dimensional fast Fourier transform (FFT).

	Input
	data  : input two-dimensional data to transform; can be a list or an array
	shift : if True, centers the zero-frequency components to the grid center
	
	Output
	freq_x   : frequency components of x-axis
	freq_y   : frequency components of y-axis
	data_fft : two-dimensional FFT
	'''

	if isinstance(data,list)==True:
		# if input data is a list, convert to array
		data = np.array(data)

	N_y,N_x = data.shape
	dt_x,dt_y = 1.,1.
	freq_x = fftpack.fftfreq(N_x,dt_x)
	freq_y = fftpack.fftfreq(N_y,dt_y)

	data_fft = fftpack.fft2(data.astype(float))

	if shift==True:
		data_fft = fftpack.fftshift(data_fft)
		freq_x   = fftpack.fftshift(freq_x)
		freq_y   = fftpack.fftshift(freq_y)

	return freq_x,freq_y,data_fft

def fIFFT2D(data,shift=True):
	'''
	Performs a two-dimensional inverse fast Fourier transform (IFFT).
	
	Input
	data  : input two-dimensional data to transform; can be a list or an array
	shift : if True, re-positions the zero-frequency components to the lower-left corner

	Output
	data_ifft : two-dimensional IFFT
	'''

	if isinstance(data,list)==True:
		# if input data is a list, convert to array
		data = np.array(data)

	if shift==True:
		data = fftpack.ifftshift(data)

	data_ifft = fftpack.ifft2(data)

	return data_ifft
