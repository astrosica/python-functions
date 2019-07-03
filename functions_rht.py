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

def fRHTdir(dir,wlen,smr,frac):
	'''
	Executes the RHT on all FITS files within a given directory.
	
	Input
	dir : the directory within which the RHT will perform.
	'''

	# get list of files within the directory
	files = os.listdir(dir)
	for file in files:
		if file[-5:]==".fits":
			filedir = dir+file
			print "Executing the RHT on file "+filedir
			rht.main(filedir,wlen=wlen,smr=smr,frac=frac)

def RHTanglediff(ijpoints_polgrad_HI,RHT_polgrad_dict,RHT_HI_dict,hthets_polgrad,hthets_HI,angles_polgrad,angles_HI):
	'''
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
		#intensity_diff_polgrad.append(intensity_polgrad)
		intensity_diff_HI.append(intensity_HI)
		# take difference between average RHT angles
		theta_diff         = thetas_polgrad_avg-thetas_HI_avg
		theta_diff_polgrad_HI.append(theta_diff)

	theta_diff_polgrad_HI = np.array(theta_diff_polgrad_HI)
	intensity_diff_HI     = np.array(intensity_diff_HI)

	return theta_diff_polgrad_HI, intensity_diff_HI

def fRHTthetadict(ijpoints,hthets):
	'''
	'''

	RHT_theta_dict = {}

	for index in range(len(ijpoints)):
		i,j=ijpoints[index]
		theta = hthets[index]
		RHT_theta_dict[i,j]=theta

	return RHT_theta_dict

def fRHTbackprojection(ipoints,jpoints,hthets,naxis1,naxis2,wlen,smr,thresh):
	'''
	Creates the RHT backprojection.
	
	Inputs
	ipoints : 
	jpoints : 
	hthets  : 
	naxis1  : 
	naxis2  : 
	wlen    : 
	smr     : 
	thresh  :
	'''

	backproj    = np.zeros(shape=(naxis2,naxis1))

	ijpoints    = zip(ipoints,jpoints)
	hthets_dict = fRHTthetadict(ijpoints,hthets)

	for ij in ijpoints:
		i,j    = ij[0],ij[1]
		hthets = hthets_dict[i,j]
		# mask hthets
		hthets_masked        = np.copy(hthets)
		hthets_masked[:15+1] = 0.0
		hthets_masked[-15:]  = 0.0
		hthets_sum           = np.sum(hthets_masked)
		backproj[j,i]        = hthets_sum

	return backproj
