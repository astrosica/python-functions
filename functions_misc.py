import os
import numpy as np
import rht, RHT_tools
import astropy.wcs as wcs
from astropy.io import fits
from PyAstronomy import pyasl
from pytpm import tpm, convert
from astropy import units as u
import matplotlib.pyplot as plt
import montage_wrapper as montage
from scipy import signal, spatial
from reproject import reproject_interp
from astropy.coordinates import SkyCoord

from matplotlib import rc
rc("text", usetex=True)

def fdegtosexa(ra_deg,dec_deg):
	'''
	Converts Right Ascension and Declination from decimal degrees to the esexagismal format.
	Inputs integers, floats, lists, or arrays.
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
	Converts Right Ascension and Declination from the sexagismal system to decimal degrees.
	Inputs integers, floats, lists, or arrays.
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

def fmaskinterp(image,mask):
	'''
	Masks and interpolates an image.

	Inputs
	image : 2D array
	mask  : 2D array of the same size as image whose masked values for invalid pixels are NaNs
	'''

	# create pixel grid
	x = np.arange(0, image.shape[1])
	y = np.arange(0, image.shape[0])
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

def fmaptheta_halfpolar_to_halfpolar(angles,deg=False):
	'''
	Maps angles from [-pi/2,pi/2) to [0,pi) or from [-90,90) to [0,180).

	Input
	angles : array of angles to be mapped
	deg    : boolean which specifies units of input angles (default unit is radian)
	'''

	if deg==False:
		# map angles within [pi,2*pi) to [0,pi)
		angles += np.pi/2.
	elif deg==True:
		# map angles within [-90,90) to [0,180)
		angles += 90.

	return angles

def fmaptheta_halfpolar(angles,deg=False):
	'''
	Maps angles from [0,2*pi) to [0,pi) or from [0,360) to [0,180).

	Input
	angles : array of angles to be mapped
	deg    : boolean which specifies units of input angles (default unit is radian)
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

def fB1950toJ2000(ra_B1950,dec_B1950):
	'''
	Transforms Right Ascension and Declination coordinates from the B1950 system to the J2000 system.
	
	Input
	ra  : list of Right Ascension coordinates in decimal degrees
	dec : list of Declination coordinates in decimal degrees
	
	Output
	ra_trans  : transformed Right Ascension coordinates in decimal degrees
	dec_trans : transformed Declination coordinates in decimal degrees
	'''
	
	def transB1950toJ2000_main(ra_J2000,dec_J2000):
		'''
		Converts Right Ascension and Declination coordinates from the sexagismal positions in the B1959 system to decimal degrees in the J2000 system.
		
		Input
		ra_J2000: single Right Ascension position in sexagismal J2000
		dec_J2000: single Declination position in sexagismal J2000
		
		Output
		ra_deg: single Right Ascension position in decimal degree
		dec_deg: single Declination position in decimal degrees
		'''
		
		# convert decimal degrees to sexagismal format
		ra_sexa,dec_sexa = degtosexa(ra_J2000,dec_J2000)
		# extract RA hh:mm:ss and Dec dd:mm:ss components
		ra_hh          = float(ra_sexa.split(" ")[0])
		ra_mm          = float(ra_sexa.split(" ")[1])
		ra_ss          = float(ra_sexa.split(" ")[2])
		dec_dd         = float(dec_sexa.split(" ")[0][1:])
		dec_mm         = float(dec_sexa.split(" ")[1])
		dec_ss         = float(dec_sexa.split(" ")[2])
		# create RA and Dec objects
		ra_J2000       = tpm.HMS(hh=ra_hh,mm=ra_mm,ss=ra_ss).to_radians()
		dec_J2000      = tpm.DMS(dd=dec_dd,mm=dec_mm,ss=dec_ss).to_radians()
		# velocity vector
		v5             = convert.cat2v6(ra_J2000,dec_J2000)
		v5_fk6         = convert.convertv6(v5,s1=5,s2=6,epoch=tpm.B1950,equinox=tpm.B1950)
		v5_fk6_ep2000  = convert.proper_motion(v5_fk6,tpm.J2000,tpm.B1950)
		d              = convert.v62cat(v5_fk6_ep2000,C=tpm.CJ)
		ra_new_rad     = d["alpha"]
		ra_deg     = ra_new_rad * 180./np.pi
		dec_new_rad    = d["delta"]
		dec_deg        = dec_new_rad * 180./np.pi
		
		return ra_deg,dec_deg
	
	ra_J2000   = []
	dec_J2000  = []
	
	if isinstance(ra_B1950,list)==True:
		'''
		If input is a list, iterate through each set of coordinates.
		'''
		for i in range(len(ra_B1950)):
			ra_i = ra_B1950[i]
			dec_i = dec_B1950[i]
			ra_new_deg,dec_new_deg=transB1950toJ2000_main(ra_i,dec_i)
			ra_J2000.append(ra_new_deg)
			dec_J2000.append(dec_new_deg)
		
		ra_J2000   = np.array(ra_J2000)
		dec_J2000  = np.array(dec_J2000)
	
	elif isinstance(ra_B1950,float)==True:
		'''
		If given a single position, transform the single set of coordinates.
		'''
		ra_new_deg,dec_new_deg=transB1950toJ2000_main(ra_B1950,dec_B1950)
		
		ra_J2000 = ra_new_deg
		dec_J2000 = dec_new_deg
	
	return ra_J2000, dec_J2000

def fmatchpos(names,ra1,dec1,ra2,dec2,minarcsec,fdir=None,fname=None,N1=None,N2=None,x1min=None,x1max=None,x2min=None,x2max=None,xlabel1=None,xlabel2=None,ylabel1=None,ylabel2=None,deg=True):
	'''
	Match two sets of pointing positions by projectings (ra1,dec1) onto (ra2,dec2).
	
	Usage :
	ra1_matches       = ra1[indices]     # matched by position
	ra1_matches_clean = ra1[indices][ii] # matched by position and cleaned by separation requirement
	ra2_clean         = ra2[ii]          # matched by position and cleaned by separation requirement
	
	Input:
	names     : IDs names of objects in second array array
	ra1       : first array of right ascension coordinates (either decimal degrees or sexagismal)
	dec1      : first array of declination coordinates (either decimal degrees or sexagismal)
	ra2       : second array of right ascension coordinates (either decimal degrees or sexagismal)
	dec2      : second array of declination coordinates (either decimal degrees or sexagismal)
	minarcsec : minimum pointing offset for matching criterium
	fdir      : output directory name for plotting offset distribution (otherwise=="None")
	fname     : output filename for plotting offset distribution (otherwise=="None")
	N1        : number of x-axis bins for plotting offset distribution (otherwise=="None")
	N2        : number of y-axis bins for plotting offset distribution (otherwise=="None")
	deg       : True if input coordinates are in decimal degrees; False if sexagismal
	
	Output:
	dist_deg_clean         : array of distances between cleaned (ra1,dec1) and (ra2,dec2) in degrees
	dist_arcsec_clean      : array of distances between cleaned (ra1,dec1) and (ra2,dec2) in arcseconds
	indices                : array of indices that match (ra1,dec1) to (ra2,dec2)
	ii                     : array of indices that clean matched (ra1,dec1) and (ra2,dec2) positions
	ii_nomatch             : array of indices that clean non-matched (ra1,dec1) positions
	ra1_deg_matches_clean  : array of ra1 positions matched to (ra2,dec2) and cleaned using minarcsec in degrees
	dec1_deg_matches_clean : array of dec1 positions matched to (ra2,dec2) and cleaned using minarcsec in degrees
	ra2_deg_clean          : array of ra2 positions matched to (ra1,dec1) and cleaned using minarcsec in degrees
	dec2_deg_clean         : array of dec2 positions matched to (ra1,dec1) and cleaned using minarcsec in degrees
	'''
	
	if deg==False:
		# convert sexagismal format to decimal degrees
		ra1_deg  = []
		dec1_deg = []
		for i in range(len(ra1)):
			ra1_i                = ra1[i]
			dec1_i               = dec1[i]
			ra1_deg_i,dec1_deg_i = sexatodeg(ra1_i,dec1_i)
			ra1_deg.append(ra1_deg_i)
			dec1_deg.append(dec1_deg_i)
		ra2_deg  = []
		dec2_deg = []
		for i in range(len(ra2)):
			ra2_i                = ra2[i]
			dec2_i               = dec2[i]
			ra2_deg_i,dec2_deg_i = sexatodeg(ra2_i,dec2_i)
			ra2_deg.append(ra2_deg_i)
			dec2_deg.append(dec2_deg_i)
	else:
		ra1_deg,dec1_deg = ra1,dec1
		ra2_deg,dec2_deg = ra2,dec2

	radec1                    = np.transpose([ra1_deg,dec1_deg])
	radec2                    = np.transpose([ra2_deg,dec2_deg])
	kdtree                    = spatial.KDTree(radec1)
	matches                   = kdtree.query(radec2)
	
	dist_deg                  = np.array(matches[0])
	dist_arcsec               = dist_deg * 3600.
	indices                   = np.array(matches[1])
	
	ra1_deg_matches           = ra1_deg[indices]
	dec1_deg_matches          = dec1_deg[indices]
	
	# matching sources
	conditions                = np.array(dist_arcsec<=minarcsec)
	ii                        = np.array(np.where(conditions)[0])
	
	dist_deg_clean            = dist_deg[ii]
	dist_arcsec_clean         = dist_arcsec[ii]
	indices_clean             = indices[ii]
	ra1_deg_matches_clean     = ra1_deg_matches[ii]
	dec1_deg_matches_clean    = dec1_deg_matches[ii]
	ra2_deg_clean             = ra2_deg[ii]
	dec2_deg_clean            = dec2_deg[ii]
	
	# non-matching sources
	conditions_nomatch        = np.array(dist_arcsec>minarcsec)
	ii_nomatch                = np.array(np.where(conditions_nomatch)[0])
	
	dist_deg_nomatch          = dist_deg[ii_nomatch]
	dist_arcsec_nomacth       = dist_arcsec[ii_nomatch]
	indices_nomatch           = indices[ii_nomatch]
	ra1_deg_matches_nomatch   = ra1_deg_matches[ii_nomatch]
	dec1_deg_matches_nomatch  = dec1_deg_matches[ii_nomatch]
	ra2_deg_nomatch           = ra2_deg[ii_nomatch]
	dec2_deg_nomatch          = dec2_deg[ii_nomatch]
	
	if fdir is not None:
		'''
		Plot resulting distribution of position offsets.
		'''
		plothist(fdir=fdir,fname=fname,hist1=dist_arcsec,N1=N1,xlabel1=xlabel1,ylabel1=ylabel1,x1min=x1min,x1max=x1max,hist2=None,N2=None,xlabel2=None,ylabel2=None,x2min=None,x2max=None,common_xaxis=False,flipx1=False,flipx2=False)
	
	return dist_deg_clean,dist_arcsec_clean,indices,ii,ii_nomatch,ra1_deg_matches_clean,dec1_deg_matches_clean,ra2_deg_clean,dec2_deg_clean
