import os
import numpy as np
import rht, RHT_tools
from astropy.io import fits
from functions_misc import fdeltatheta

def fRHTdir(dir,wlen,smr,frac):
	'''
	Executes the RHT on all FITS files within a given directory.
	
	Input
	dir  : the directory within which the RHT will perform
	wlen : the window length (D_W) which sets the length of a structure (default=21)
	smr  : the unsharp mask smoothing radius (D_K; default=2)
	frac : fraction/percent of one angle that must be 'lit up' to be counted (default=0.7)

	Output
	saves the RHT FITS table to the input dir directory
	'''

	# get list of files within the directory
	files = os.listdir(dir)
	for file in files:
		if file[-5:]==".fits":
			filedir = dir+file
			print("Executing the RHT on file "+filedir)
			rht.main(filedir,wlen=wlen,smr=smr,frac=frac)

def faddtoRHTheader(rhtfile,newheader):
	'''
	Adds a FITS header to the RHT FITS table.

	Input
	rhtfile   : file directory to RHT file
	newheader : header to be saved to RHT file
	'''

	# extract RHT data
	hdu      = fits.open(rhtfile,mode="readonly",memmap=True,save_backup=False,checksum=True)
	priHDU   = hdu[0]        # primary HDU
	tblHDU   = hdu[1]        # table HDU
	prihdr   = priHDU.header # primary header
	backproj = priHDU.data   # primary data (backprojection)

	# collect FITS header keys
	newkeys = newheader.keys()
	oldkeys = prihdr.keys()

	# add keys from new header
	for key in newkeys:
		if key not in oldkeys:
			try:
				prihdr[str(key)] = newheader[str(key)]
			except ValueError:
				continue

	# over-write file
	prihdu   = fits.PrimaryHDU(data=backproj,header=prihdr) # primary HDU
	thdulist = fits.HDUList([priHDU,tblHDU])                # table HDUlist
	thdulist.writeto(rhtfile,output_verify="silentfix",overwrite=True,checksum=True)

def RHTanglediff(ijpoints_polgrad_HI,RHT_polgrad_dict,RHT_HI_dict,hthets_polgrad,hthets_HI,angles_polgrad,angles_HI):
	'''
	Computes the RHT angle difference between two maps.

	Input
	ijpoints_polgrad_HI : list or array of pixel positions (i,j)
	RHT_polgrad_dict    : dictionary of RHT angles for each pixel position (i,j)
	RHT_HI_dict         : dictionary of RHT angles for each pixel position (i,j)
	hthets_polgrad      : list or array of RHT angle intensities
	hthets_HI           : list or array of RHT angle intensities
	angles_polgrad      : list or array of RHT angles
	angles_HI           : list or array of RHT angles

	Output
	theta_diff_polgrad_HI : array of RHT angle differences
	intensity_diff_HI     : array of intensities 
	'''
	theta_diff_polgrad_HI  = []
	intensity_diff_HI      = []

	# iterate through spatial pixels common to non-zero polarization gradient and HI RHT backprojections
	for pixel in ijpoints_polgrad_HI:
		# collect RHT hthets arrays for each spatial pixel
		hthets_polgrad     = RHT_polgrad_dict[pixel]
		hthets_HI          = RHT_HI_dict[pixel]
		# find RHT angles with non-zero intensities
		thetas_polgrad     = angles_polgrad[np.where(hthets_polgrad>0)[0]]
		thetas_HI          = angles_HI[np.where(hthets_HI>0)[0]]
		# take averages of non-zero RHT angles
		thetas_polgrad_avg = np.mean(thetas_polgrad)
		thetas_HI_avg      = np.mean(thetas_HI)
		# find RHT intensities at each spatial pixel
		intensity_polgrad  = np.sum(hthets_polgrad)
		intensity_HI       = np.sum(hthets_HI)
		intensity_diff_polgrad.append(intensity_polgrad)
		intensity_diff_HI.append(intensity_HI)
		# take difference between average RHT angles
		theta_diff         = thetas_polgrad_avg-thetas_HI_avg
		theta_diff_polgrad_HI.append(theta_diff)

	theta_diff_polgrad_HI  = np.array(theta_diff_polgrad_HI)
	intensity_diff_HI      = np.array(intensity_diff_HI)

	return theta_diff_polgrad_HI, intensity_diff_HI

def fRHTargdiff(ijpoints,arg_dict,RHT_dict,angles):
	'''
	Computes the angle difference between the polarization gradient argument and the HI RHT.

	Input
	ijpoints : list or array of pixel positions (i,j)
	arg_dict : dictionary of polarization gradient argument for each pixel position (i,j) in degrees on [-90,+90]
	RHT_dict : dictionary of RHT angle intensities for each pixel position (i,j)
	angles   : list or array of RHT angles associated with hthets in degrees on [0,180)

	Output
	theta_diff : array of differences between arguments of polarizartion gradients and HI fibers
	'''

	theta_diff = []
	# 0 -- 16 degrees    : ignore
	# 164 -- 180 degrees : ignore
	# 16 -- 164 degrees  : want

	#angles_deg = np.degrees(angles)

	# iterate through spatial pixels common to non-zero polarization gradient and HI RHT angle intensities
	for ijpoint in ijpoints:
		arg_ij       = arg_dict[ijpoint]                # single angle
		hthets_ij    = RHT_dict[ijpoint]                # list of intensities as a function of angle
		# extract non-zero intensity RHT angles
		thetas_ij    = angles[np.where(hthets_ij>0)[0]] # list of angles
		# iterate through RHT angles and compute difference from argument
		if (arg_ij!=np.nan) and (np.isnan(arg_ij)==False):
			# ignore nan argument angles
			for theta_ij in thetas_ij:
				# ignore artefact RHT angles
				if (theta_ij>16.) and (theta_ij<164.):
					#arg_RHT_diff = np.abs((arg_ij+90.)) - np.abs((theta_ij))
					arg_RHT_diff = fdeltatheta(arg_ij,theta_ij,"deg","deg")
					theta_diff.append(arg_RHT_diff)

	theta_diff = np.array(theta_diff)

	return theta_diff

def fRHTargdiff_filament_map(ijpoints,arg_dict,RHT_dict,angles):
	'''
	Computes a map of the the angular offset between the polarization gradient argument and the HI RHT for the bright polarized filament in N1.

	Input
	ijpoints : list or array of pixel positions (i,j)
	arg_dict : dictionary of polarization gradient argument for each pixel position (i,j) in degrees on [-90,+90]
	RHT_dict : dictionary of RHT angle intensities for each pixel position (i,j)
	angles   : list or array of RHT angles associated with hthets in degrees on [0,180)

	Output
	theta_diff : array of differences between arguments of polarizartion gradients and HI fibers
	'''

	theta_diff_map = np.zeros(shape=(455, 455))
	# 0 -- 16 degrees    : ignore
	# 164 -- 180 degrees : ignore
	# 16 -- 164 degrees  : want

	#angles_deg = np.degrees(angles)

	# iterate through spatial pixels common to non-zero polarization gradient and HI RHT angle intensities
	for ijpoint in ijpoints:
		arg_ij       = arg_dict[ijpoint]                # single angle
		hthets_ij    = RHT_dict[ijpoint]                # list of intensities as a function of angle
		# extract non-zero intensity RHT angles
		thetas_ij    = angles[np.where(hthets_ij>0)[0]] # list of angles
		# iterate through RHT angles and compute difference from argument
		if (arg_ij!=np.nan) and (np.isnan(arg_ij)==False):
			# ignore nan argument angles
			for theta_ij in thetas_ij:
				# ignore artefact RHT angles
				if (theta_ij>16.) and (theta_ij<164.):
					arg_RHT_diff = fdeltatheta(arg_ij,theta_ij,"deg","deg")
					theta_diff_map[ijpoint[1],ijpoint[0]] = arg_RHT_diff

	return theta_diff_map

def fRHTthetadict(ijpoints,hthets):
	'''
	Creates a dictionary of RHT angle intensities for each pixel in the image plane.

	Input 
	ijpoints : list or array of pixel positions (i,j)
	hthets   : list or array of RHT angles

	Output
	RHT_theta_dict : dictionary of RHT angles for each pixel position (i,j)
	'''

	RHT_theta_dict = {}

	for index in range(len(ijpoints)):
		i,j=ijpoints[index]
		theta = hthets[index]
		RHT_theta_dict[i,j]=theta

	return RHT_theta_dict

def fRHTbackprojection(ipoints,jpoints,hthets,naxis1,naxis2):
	'''
	Creates the RHT backprojection.
	
	Inputs
	ipoints : y- pixel positions in the image plane
	jpoints : x- pixel positions in the image plane
	hthets  : RHT angle intensities
	naxis1  : length of image x-axis
	naxis2  : length of image y-axis

	Output
	backproj : RHT backprojection
	'''

	backproj    = np.zeros(shape=(naxis2,naxis1))

	ijpoints    = zip(ipoints,jpoints)
	hthets_dict = fRHTthetadict(ijpoints,hthets)

	for ij in ijpoints:
		i,j    = ij[0],ij[1]
		hthets = hthets_dict[i,j]
		# mask hthets
		hthets_masked        = np.copy(hthets)
		# polarization gradient
		#hthets_masked[:15+1] = 0.0
		#hthets_masked[-15:]  = 0.0
		# weight map FFT
		hthets_masked[0]     = 0.0
		hthets_sum           = np.sum(hthets_masked)
		backproj[j,i]        = hthets_sum

	return backproj
