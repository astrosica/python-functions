import os
import numpy as np
from scipy import fftpack
from scipy import interpolate
from scipy import signal, spatial
from astropy import constants as const
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve_fft

from matplotlib import rc
rc("text", usetex=True)

def fwavel(freq):
	'''
	Connverts frequency to wavelength.

	Input
	freq : frequency [Hz]

	Output
	wavel : wavelength [m]
	'''

	wavel = const.c.value/freq

	return wavel

def ffreq(wavel):
	'''
	Converts wavelength to frequency.

	Input
	wavel : wavelength [m]

	Output
	freq : frequency [Hz]
	'''

	freq = const.c.value/wavel

	return freq

def fTk(delta_v,delta_t=None):
	'''
	Computes the upper-limit on kinetic temperature from line broadening.

	Input
	delta_v : line width in velocity [km/s]

	Output
	Tk : kinetic temperature	
	'''

	mH = 1.00784*const.u.value
	kB = const.k_B.value

	if delta_t is None:
		# only thermal broadening
		num = mH*delta_v**2.
		den = 3.*kB
		Tk  = num/den
	elif delta_t is not None:
		# includes turbulent broadening
		num = mH*(delta_v**2. - delta_t**2.)
		den = 3.*kB
		Tk  = num/den

	return Tk

def fmask_signal(data,signal,fill_value=np.nan,lessthan=True):
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
	mask = np.ones(shape=data.shape) # initialize mask
	if lessthan is True:
		low_signal = np.where(data<signal)
	elif lessthan is not True:
		low_signal = np.where(data>signal)
	mask[low_signal] = fill_value
	
	# mask data
	data_clean       = data*mask

	return (mask,data_clean)

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

	N_y,N_x   = data.shape
	dt_x,dt_y = 1.,1.
	freq_x    = fftpack.fftfreq(N_x,dt_x)
	freq_y    = fftpack.fftfreq(N_y,dt_y)

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

def fdeltatheta(theta1,theta2,inunit,outunit):
	'''
	Computes the angular difference between two angle measurements defined on the half-polar plane [0,180).
	See Equation 15 in Clark & Hensley (2019)

	Input
	theta1 : list or array of reference angles
	theta2 : list or array of angles whose offsets from the reference angles we want to know
	inunit : units of angular inputs
	outunit : units of angular offset outputs

	Output
	delta_theta : array of angular offsets in units of outunit
	'''

	degree_units = ["deg","degree","degrees"]
	rad_units    = ["rad","radian","radians"]

	if inunit in degree_units:
		theta1_rad = np.radians(np.array(theta1))
		theta2_rad = np.radians(np.array(theta2))
	elif inunit in rad_units:
		theta1_rad = np.array(theta1)
		theta2_rad = np.array(theta2)

	num = np.sin(2.*theta1_rad)*np.cos(2.*theta2_rad) - np.cos(2.*theta1_rad)*np.sin(2.*theta2_rad)
	den = np.cos(2.*theta1_rad)*np.cos(2.*theta2_rad) + np.sin(2.*theta1_rad)*np.sin(2.*theta2_rad)

	delta_theta = 0.5 * np.arctan2(num,den) # angle measured on [-pi,+pi] in radians

	if outunit in degree_units:
		delta_theta = np.degrees(delta_theta)

	return delta_theta

def fconvolve(oldres,newres,data,header,restype="FWHM",method="scipy"):
	'''
	Convolves an image using FFT convolution.
	Note: newres is not the size of the convolution kernel, this will be solved for.
	
	Input
	oldres  : input resolution in arcminutes (FWHM)
	newres  : desired resolution in arcminutes (FWHM)
	data    : image to be convolved
	header  : FITS header of image
	restype : type of resolution; can be either sigma (standard deviation) or FWHM (default=sigma)
	method  : method of interpolation; can be scipy or astropy (default=scipy)
	
	Output
	data_smoothed : smoothed image
	'''
	
	if restype=="sigma":
		# do nothing
		oldres_sigma = oldres
		newres_sigma = newres
	if restype=="FWHM":
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

def freadROHSA(filedir,shape,vmin,vmax,deltav):
	'''
	Reads in a dat file containing ROHSA results and outputs the corresponding data.
	See Marchal et al. (2019) for a description of ROHSA.
	https://github.com/antoinemarchal/ROHSA

	Input
	filedir : directory to ROHSA dat file
	ny      : nunmber of pixels in y-dimension
	nx      : number of pixels in x-dimension

	Output
	amplitude : amplitude of Gaussian components         [pixels]
	position : central position of Gaussian components   [pixels]
	dispersion : dispersion of Gaussian componennts      [pixels]
	'''

	def gauss_2D(xs,a,mu,sig):
		return [a * np.exp(-((x - mu)**2)/(2. * sig**2)) for x in xs]

	data = np.genfromtxt(filedir)
	dim_y,dim_x = shape[1],shape[2]

	amp = data[:, 2]
	mean = data[:, 3] - 1
	sigma = data[:, 4]

	n_gauss = int(len(amp) / (dim_y*dim_x))
	params = np.zeros((3*n_gauss, dim_y, dim_x))

	i__ = 0
	for i in range(dim_y):
		for j in range(dim_x):
			for k in range(n_gauss):
				params[0+(3*k),i,j] = amp[i__]
				params[1+(3*k),i,j] = mean[i__]
				params[2+(3*k),i,j] = sigma[i__]
				i__ += 1

	amplitude  = params[0::3]
	position   = params[1::3]
	dispersion = params[2::3]

	model = np.zeros(shape)
	n_gauss = params.shape[0]/3

	for i in np.arange(n_gauss):
		model += gauss_2D(np.arange(shape[0]),params[int(0+(3*i))],params[int(1+(3*i))],params[int(2+(3*i))])

	# translate amplitude to column density (assuming optically thin emission)
	columnden = amplitude*1.823E18

	# translate pixel axis to velocity axis
	vel_axis = np.arange(vmin,vmax,deltav)
	pix_axis = np.arange(len(vel_axis))
	fvel     = interpolate.interp1d(pix_axis,vel_axis)
	velocity = fvel(position)

	# translate dispersion from pixel to velocity units
	dispersion_v = dispersion*deltav

	#return amplitude,position,dispersion,model
	return amplitude,columnden,position,velocity,dispersion,dispersion_v,model

def fmaskpointsources(image,xpoints,ypoints,radius,interp="cubic"):
	'''
	Masks background point sources and interpolates over them.
	Input
	image   : two-dimensional image to be masked
	xpoints : x pixel positions of sources to be masked
	ypoints : y pixel positions of sources to be masked
	radius  : mask radius
	interp  : type of interpolation (default=cubic)
	Output
	image_masked_interp : masked and interpolated image
	'''

	# initialize mask
	mask = np.ones(shape=image.shape).astype("bool") # initialize mask

	for i in range(len(xpoints)):
		x0,y0    = xpoints[i],ypoints[i]
		mask_i,_ = fmask_circle(image,x0,y0,radius)
		# iteratively adjust mask
		mask *=mask_i

	# mask image
	image_masked = image*mask

	# interpolate over mask
	image_masked_interp = fmaskinterp(image_masked,mask)

	return image_masked_interp,mask

def fmaskinterp(image,mask):
	'''
	Masks and interpolates a two-dimensional image.

	Input
	image : 2D array
	mask  : 2D array of the same size as image whose masked values for invalid pixels are NaNs

	Output
	image_interp : the masked and interpolated image
	'''

	# create pixel grid
	x      = np.arange(0, image.shape[1])
	y      = np.arange(0, image.shape[0])
	xx, yy = np.meshgrid(x,y)

	# mask image
	image_masked    = image*mask
	# replace False with Nans
	image_masked[~mask] = np.nan
	masked_mask_obj     = np.ma.masked_invalid(image_masked)

	#get only the valid values
	x1        = xx[~masked_mask_obj.mask]
	y1        = yy[~masked_mask_obj.mask]
	image_new = image[~masked_mask_obj.mask]

	# interpolate
	image_interp = interpolate.griddata((x1, y1), image_new.ravel(),(xx, yy),method="nearest")

	return image_interp

def fmask_circle(image,x0,y0,r):
	'''
	Masks an image within the boundaries of a circle.

	Input
	image  : image to be masked
	x0     : x-coordinate of circle center
	y0     : y-coordinates of circle center
	r      : circle radius

	Output
	mask         : output mask
	image_masked : masked image
	'''

	NAXIS2,NAXIS1 = image.shape
	xpix,ypix     = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid   = np.meshgrid(xpix,ypix)

	mask          = r**2. > ((xgrid-x0)**2.) + ((ygrid-y0)**2.)
	mask          = np.invert(mask)

	image_masked  = image*mask

	return mask,image_masked

def fmask_ellipse(image,x0,y0,r,a,b):
	'''
	Masks an image within the boundaries of an ellipse.

	Input
	image  : image to be masked
	x0     : x-coordinate of circle center
	y0     : y-coordinates of circle center
	r      : circle radius
	a      : horizontal stretch (a>0) or compression (a<0)
	b      : vertical stretch (b>0) or compression (b<0)

	Output
	mask         : output mask
	image_masked : masked image
	'''

	NAXIS2,NAXIS1 = image.shape
	xpix,ypix     = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid   = np.meshgrid(xpix,ypix)

	mask          = r**2. > ((xgrid-x0)**2.)/a + ((ygrid-y0)**2.)/b
	mask          = np.invert(mask)

	image_masked  = image*mask

	return mask,image_masked

def fmask_slab(image,theta1,theta2,x0_1,y0_1,x0_2,y0_2,scale_x,scale_y,angleunits="deg"):
	'''
	'''

	NAXIS2,NAXIS1      = image.shape
	xpix,ypix          = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid        = np.meshgrid(xpix,ypix)

	deg_units = ["deg","degree","degrees"]
	rad_units = ["rad","radian","radians"]

	if angleunits in deg_units:
		deltax_1, deltay_1 = scale_x*np.cos(np.radians(theta1)), scale_y*np.sin(np.radians(theta1))
		deltax_2, deltay_2 = scale_x*np.cos(np.radians(theta2)), scale_y*np.sin(np.radians(theta2))
	else:
		deltax_1, deltay_1 = scale_x*np.cos(theta1), scale_y*np.sin(theta1)
		deltax_2, deltay_2 = scale_x*np.cos(theta2), scale_y*np.sin(theta2)
	slope_1,slope_2    = deltay_1/deltax_1, deltay_2/deltax_2
	b_1,b_2            = (y0_1-slope_1*x0_1, y0_2-slope_2*x0_2)
	y_1,y_2            = (slope_1*xgrid+b_1, slope_2*xgrid+b_2)

	mask               = (ygrid>y_1) & (ygrid<y_2)
	mask               = np.invert(mask)

	image_masked       = image*mask

	return mask,image_masked

def fmask_sensitivitymap(image):
	'''
	Creates a mask for Arecibo's sensitivity map structures.

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
	image_masked = image*mask

	return mask,image_masked

def fmask_basketweaving_diffuse(image,pointing):
	'''
	Creates a mask for Arecibo's basketweaving artefacts in Fourier space towards a diffuse region.

	Input
	image    : two-dimensional data to be masked (FFT of sensitivity map)
	pointing : GALFACTS pointing (N[1-4],S[1-4])
	
	Output
	mask         : resulting mask
	image_masked : masked image
	'''

	NAXIS2,NAXIS1 = image.shape
	xpix,ypix     = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid   = np.meshgrid(xpix,ypix)

	if pointing=="N1":
		xypoints = np.array([
			[0,618,1073,441,0,633,1073,455],
			[0,443,1073,617,0,457,1073,631]
			])
	elif pointing=="N3":
		xypoints = np.array([
			[0,268,472,190,0,282,472,204],
			[0,190,472,271,0,204,472,285]
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

	if pointing=="N1":
		mask[:,492:582] = True
	elif pointing=="N3":
		mask[:,223:250] = True

	# convert mask to ones and zeros
	mask = mask.astype(float)

	# mask image
	image_masked = image*mask

	return mask,image_masked

def fmask_basketweaving_filament(image):
	'''
	Creates a mask for Arecibo's basketweaving artefacts in Fourier space towards a polarized filament.

	Input
	image : two-dimensional data to be masked (FFT of sensitivity map)
	
	Output
	mask         : resulting mask
	image_masked : masked image
	'''

	NAXIS2,NAXIS1 = image.shape
	xpix,ypix     = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid   = np.meshgrid(xpix,ypix)

	# N1
	xypoints = np.array([
		[0,257,454,177,0,275,454,195],
		[0,180,454,256,0,198,454,274]
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

	# N1
	mask[:,202:252]   = True
	
	# convert mask to ones and zeros
	mask = mask.astype(float)

	# mask image
	image_masked = image*mask

	return mask,image_masked

def fmask_basketweaving(image,pointing):
	'''
	Creates a mask for Arecibo's basketweaving artefacts in Fourier space.

	Input
	image    : two-dimensional data to be masked (FFT of sensitivity map)
	pointing : GALFACTS pointing [N[1-4],S[1-4]]
	
	Output
	mask         : resulting mask
	image_masked : masked image
	'''

	NAXIS2,NAXIS1 = image.shape
	xpix,ypix     = np.arange(0,NAXIS1),np.arange(0,NAXIS2)
	xgrid,ygrid   = np.meshgrid(xpix,ypix)

	if pointing=="N1" or pointing=="N2" or pointing=="N4" or pointing=="S1" or pointing=="S2" or pointing=="S3" or pointing=="S4":
		xypoints = np.array([
			[0,575,5369,507,0,692,5369,380], # - + + - 
			[0,384,5369,688,0,511,5369,561], # - + + - 
			[0,692,5369,380,0,565,5369,507], # + - - +
			[0,511,5369,561,0,384,5369,688]  # + - - +
			])
	elif pointing=="N3":
		xypoints = np.array([
		[0,575,5369,507,0,692,5369,380],     # - + + - 
			[0,384,5369,688,0,511,5369,561], # - + + - 
			[0,692,5369,380,0,565,5369,507], # + - - +
			[0,511,5369,561,0,384,5369,688]  # + - - +
			])
	
	# initialize mask
	mask = np.full((NAXIS2,NAXIS1),True,dtype=bool)

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
	if pointing=="N1":
		#mask[:,2461:2918] = True
		mask[:,2681:2698] = True
	elif pointing=="N2":
		mask[:,2656:2896] = True
	elif pointing=="N3":
		mask[:,2959:3223] = True
	elif pointing=="N4":
		mask[:,2762:3088] = True
	elif pointing=="S1":
		mask[:,2423:2733] = True
	elif pointing=="S2":
		mask[:,2424:2978] = True
	elif pointing=="S3":
		mask[:,2500:3022] = True
	elif pointing=="S4":
		mask[:,2583:2819] = True	

	# convert mask to ones and zeros
	mask = mask.astype(float)

	# mask image
	image_masked = image*mask

	return mask,image_masked

