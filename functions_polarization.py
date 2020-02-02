import sys
import aplpy
import numpy as np
import astropy.wcs as wcs
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy import constants as const
from matplotlib.collections import LineCollection

from functions_misc import fmask_signal

def fPI(Q,U):
	'''
	Coomputes the polarized intensity.
	
	Input
	Q  : Stokes Q map
	U  : Stokes U map
	
	Output
	PI : polarized intensity
	'''
	
	# compute polarized intensity
	PI = np.sqrt(Q**2.+U**2.)
	
	return PI

def fpolfrac(I,Q,U):
	'''
	Coomputes the polarization fraction.
	
	Input
	I : Stokes I map
	Q : Stokes Q map
	U : Stokes U map
	
	Output
	polfrac : polarization fraction
	'''
	
	# compute polarized intensity
	PI = np.sqrt(Q**2.+U**2.)

	# compute the polarization fraction
	polfrac = PI/I
	
	return polfrac

def fPI_debiased(Q,U,Q_std,U_std):
	'''
	Compute the de-biased polarized intensity.
	
	Input
	Q     : Stokes Q map
	U     : Stokes U map
	Q_std : standard deviation of Stokes Q noise
	U_std : standard deviation of Stokes U noise

	Output
	PI_debiased : debiased polarized intensity
	'''

	# compute effective Q/U noise standard deviation
	std_QU = np.sqrt(Q_std**2. + U_std**2.)

	# compute polarized intensity
	PI = np.sqrt(Q**2.+U**2.)
	# compute de-biased polarized intensity
	PI_debiased = PI * np.sqrt( 1. - (std_QU/PI)**2. )
	
	return PI_debiased

def fpolgrad(Q,U):
	'''
	Computes the polarization gradient.
	See Equation 1 in Gaensler et al. (2011).
	
	Input
	Q : Stokes Q map
	U : Stokes U map

	Output
	polgrad : polarization gradient
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)
	
	# compute spatial polarization gradient
	polgrad  = np.sqrt(Q_grad_x**2.+Q_grad_y**2.+U_grad_x**2.+U_grad_y**2.)
	
	return polgrad

def fpolgradnorm(Q,U):
	'''
	Computes the normalized polarization gradient.
	See Iacobelli et al. (2014).
	
	Input
	Q : Stokes Q map
	U : Stokes U map
	
	Output
	polgrad_norm : normalized polarization gradient
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)
	
	# compute spatial polarization gradient
	polgrad  = np.sqrt(Q_grad_x**2.+Q_grad_y**2.+U_grad_x**2.+U_grad_y**2.)
	
	# compute the polarized intensity
	P = np.sqrt(Q**2.+U**2.)
	
	# compute normalized polarization gradient
	polgrad_norm = polgrad/P
	
	return polgrad_norm

def fpolgrad_crossterms(Q,U):
	'''
	Computes the polarization gradient with cross-terms.
	See Equation 15 in Herron et al. (2018).
	
	Input
	Q : Stokes Q data
	U : Stokes U data

	Output
	polgrad : polarization gradient with cross-terms
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)
	
	# compute spatial polarization gradient
	a       = Q_grad_x**2.+Q_grad_y**2.+U_grad_x**2.+U_grad_y**2.
	b       = a**2. - 4.*(Q_grad_x*U_grad_y - Q_grad_y*U_grad_x)**2.
	
	# compute polarization gradient
	polgrad = np.sqrt(0.5*a + 0.5*np.sqrt(b))
	
	return polgrad

def fpolgradarg(Q,U,parallel=False,deg=True):
	'''
	Computes the argument of the polarization gradient.
	See the equation in the caption of Figure 2 in Gaensler et al. (2011).
	
	Input
	Q        : Stokes Q data
	U        : Stokes U data
	parallel : if True, compute angle parallel (rather then perpendicular) to polarization gradient structures (default=False)
	deg      : if True, converts the argument to degrees for output

	Output
	polgrad_arg : argument of polarization gradient
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)

	# compute argument of polarization gradient
	a = np.sign(Q_grad_x*Q_grad_y + U_grad_x*U_grad_y)
	b = np.sqrt(Q_grad_y**2.+U_grad_y**2.)
	c = np.sqrt(Q_grad_x**2.+U_grad_x**2.)

	polgrad_arg = np.arctan(a*b/c) # angle measured from the x-axis on [-pi/2,+pi/2] in radians

	if parallel==True:
		# compute argument angle parallel to filaments from North (like the RHT)
		polgrad_arg += np.pi/2. # angle measured from the y-axis on [0,pi] in radians

	if deg==True:
		# convert to degrees
		polgrad_arg = np.degrees(polgrad_arg)

	return polgrad_arg

def fpolgradarg_crossterms(Q,U,parallel=False,deg=True):
	'''
	Computes the argument of the polarization gradint with cross-terms.
	See Equations 13 and 14 in Herron et al. (2018).
	
	Input
	Q        : Stokes Q map
	U        : Stokes U map
	parallel : if True, compute angle parallel (rather then perpendicular) to polarization gradient structures
	deg      : if True, converts to degrees at the end
	
	Output
	polgrad_arg : argument of polarization gradient
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)

	# compute the cos(2*theta) term
	cos2theta_num = -(Q_grad_y**2. - Q_grad_x**2. + U_grad_y**2. - U_grad_x**2.)
	cos2theta_den = np.sqrt((Q_grad_x**2. + Q_grad_y**2. + U_grad_x**2. + U_grad_y**2.)**2. - 4.*(Q_grad_x*U_grad_y - Q_grad_y*U_grad_x)**2.)
	cos2theta     = cos2theta_num/cos2theta_den

	# compute the sin(2*theta) term
	sin2theta_num = 2.*(Q_grad_x*Q_grad_y + U_grad_x*U_grad_y)
	sin2theta_den = np.sqrt((Q_grad_x**2. + Q_grad_y**2. + U_grad_x**2. + U_grad_y**2.)**2. - 4.*(Q_grad_x*U_grad_y - Q_grad_y*U_grad_x)**2.)
	sin2theta     = sin2theta_num/sin2theta_den

	# compute tan(theta)
	tantheta_num  = sin2theta
	tantheta_den  = 1.+cos2theta
	# take inverse tan to compute argument
	polgrad_arg   = np.arctan2(tantheta_num,tantheta_den) # angle measured from the x-axis on [-pi,+pi] in radians
	# transform angles from [-pi,pi] to [-pi/2,pi/2]
	polgrad_arg[polgrad_arg<-np.pi/2.] += np.pi
	polgrad_arg[polgrad_arg>np.pi/2.]  += np.pi

	if parallel==True:
		# compute argument angle parallel to filaments from North (like the RHT)
		polgrad_arg += np.pi/2. # angle measures from the y-axis on [0,pi] in radians

	if deg==True:
		# convert to degrees
		polgrad_arg = np.degrees(polgrad_arg)

	return polgrad_arg

def fargmask(angles,min,max):
	'''
	Creates a mask for the argument of polarization gradient based on an input of angle range(s).

	Inputs
	angles : angle map
	min    : minimum of the range of angles to be masked (can be single-valued or a list/array)
	max    : maximum of the range of angles to be masked (can be single-valued or a list/array)

	Output
	mask : a mask the same size as the angle map
	'''

	# initialize mask
	mask = np.ones(shape=angles.shape)

	# fill in mask using input angles
	for i in range(len(min)):
		mask_i              = np.copy(mask)
		min_i               = min[i]
		max_i               = max[i]
		mask_angles         = np.where((angles>=min_i) & (angles<=max_i))
		mask_i[mask_angles] = np.nan
		mask               *=mask_i

	return mask

def fpolgradnorm_crossterms(Q,U):
	'''
	Computes the complete normalized polarization gradient with cross-terms.
	
	Input
	Q : Stokes Q data
	U : Stokes U data

	Output
	polgrad_norm : normalized polarization gradient with cross-terms
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)
	
	# compute spatial polarization gradient
	a       = Q_grad_x**2.+Q_grad_y**2.+U_grad_x**2.+U_grad_y**2.
	b       = a**2. - 4.*(Q_grad_x*U_grad_y - Q_grad_y*U_grad_x)**2.
	polgrad = np.sqrt(0.5*a + 0.5*np.sqrt(b) )
	
	# compute the polarized intensity
	P = np.sqrt(Q**2.+U**2.)
	
	# compute the normalized polarization gradient
	polgrad_norm = polgrad/P
	
	return polgrad_norm

def fpolgrad_rad(Q,U):
	'''
	Computes the radial component of the polarization gradient.
	See Equation 22 in Herron et al. (2018).
	
	Input
	Q : Stokes Q map
	U : Stokes U map

	Output
	polgrad_rad : radial component of the polarization gradient
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)
	
	polgrad_rad_num = (Q*Q_grad_x+U*U_grad_x)**2. + (Q*Q_grad_y+U*U_grad_y)**2.
	polgrad_rad_den = Q**2.+U**2.
	
	# compute radial component of polarization gradient
	polgrad_rad = np.sqrt(polgrad_rad_num/polgrad_rad_den)
	
	return polgrad_rad
    
def fpolgrad_tan(Q,U):
	'''
	Computes the radial component of the polarization gradient.
	See Equation 25 in Herron et al. (2018).
	
	Input
	Q : Stokes Q map
	U : Stokes U map

	Output
	polgrad_tan : tangential component of the polarization gradient
	'''
	
	# compute Stokes spatial gradients
	Q_grad_y,Q_grad_x = np.gradient(Q)
	U_grad_y,U_grad_x = np.gradient(U)
	
	polgrad_tan_num = (Q*U_grad_x+U*Q_grad_x)**2. + (Q*U_grad_y-U*Q_grad_y)**2.
	polgrad_tan_den = Q**2.+U**2.
	
	# compute tangential component of polarization gradient
	polgrad_tan = np.sqrt(polgrad_tan_num/polgrad_tan_den)
	
	return polgrad_tan

def fpolangle(Q,U,toIAU=False,deg=True):
	'''
	Computes the polarization angle.
	
	Input
	Q   : Stokes Q
	U   : Stokes U
	toIAU : if True, converts from COSMO to IAU convention (default=False)
	deg : if True, convert angles to degrees for output (default=True)

	Output
	polangle : polarization angle
	'''

	if toIAU==True:
		# if converting from COSMOS to IAU convention
		# flip sign of Stokes U
		U *= -1.

	# polarization angle
	pol_angle = np.mod(0.5*np.arctan2(U,Q), np.pi)

	if deg==True:
		# convert from radians to degrees
		pol_angle = np.degrees(pol_angle)

	return pol_angle

def fBangle(Q,U,toIAU=False,deg=True):
	'''
	Computes the plane-of-sky magnetic field angle.
	
	COSMO polarization angle convention to IAU B-field convention = flip sign of Stokes U
	
	Input
	Q     : Stokes Q
	U     : Stokes U
	toIAU : if True, converts from COSMO to IAU convention (default=False)
	deg   : if True, convert angles to degrees for output (default=True)

	Output
	polangle : magnetic field angle
	'''

	# flip both Q and U to convert from electric
	# field to magnetic field (equivalent to a
	# 90 degree rotation)

	Q *= -1.
	U *= -1.

	if toIAU==True:
		# if converting from COSMOS to IAU convention
		# flip sign of Stokes U
		U *= -1.

	# magnetic field angle
	B_angle = np.mod(0.5*np.arctan2(U,Q), np.pi)

	if deg==True:
		# convert from radians to degrees
		B_angle = np.degrees(B_angle)

	return B_angle

def fderotate(pangle,RM,wavel,inunit,outunit):
	'''
	Computes the de-rotated polarization angles.

	Input
	pangle  : polarization angle                 [degrees or radians]
	RM      : rotation measure                   [rad/m^2]
	wavel   : observing wavelength               [m]
	inunit  : units of input polarization angle  [degrees or radians]
	outunit : units of output polarization angle [degrees or radians]
	'''

	degree_units = ["deg","degree","degrees"]
	rad_units    = ["rad","radian","radians"]

	if inunit in degree_units:
		# if input polarization angle is in degrees
		pangle_deg = pangle
		pangle_rad = np.radians(pangle)
	elif inunit in rad_units:
		# if input polarization angle is in radians
		pangle_rad = pangle
		pangle_deg = np.degrees(pangle)

	pangle_0_rad = np.mod(pangle_rad-RM*wavel**2.,np.pi)
	pangle_0_deg = np.degrees(pangle_0_rad)

	if outunit in degree_units:
		# if output polarization angle is in degrees
		pangle_0 = pangle_0_deg
	elif outunit in rad_units:
		# if output polarization angle is in radians
		pangle_0 = pangle_0_rad

	return pangle_0

def fPlanckMJy(I,Q,U):
	'''
	Converts Planck Stokes maps from K_CMB to astrophysical units MJy/sr.
	'''

	fac = 287.5

	I_MJysr = I*fac
	Q_MJysr = Q*fac
	U_MJysr = U*fac

	return I_MJysr,Q_MJysr,U_MJysr

def fPlanckCMBmonopole(data,unit="K"):
	'''
	Removes the CMB monopole from the Planck 353 GHz Stokes maps.
	Offset corrections from Planck III (2018).

	Input
	data : Planck Stokes map
	unit : dimensions of Stokes map [K or MJysr]
	'''

	if unit=="K":
		offset = 452E-6
	elif unit=="MJysr":
		offset = 0.13

	data_corr = data - offset

	return data_corr

def fPlanckHIcorr(data,unit="K"):
	'''
	Adds the Galactic HI offset correction to the Planck 353 GHz Stokes I map.
	Offset correction from Planck XII (2018).

	Input
	data : Planck Stokes I map
	unit : dimensions of Stokes map [K or MJysr]
	'''

	if unit=="K":
		offset = 36E-6
	elif unit=="MJysr":
		fac = 287.5 # converts K_CMB to MJy/sr
		offset = 36E-6 * fac

	data_corr = data + offset

	return data_corr

def fpolgradargdict(polgrad_arg):
	'''
	Creates a dictionary of polarization gradient arguments for each pixel in the image plane.
	
	Input
	polgrad_arg : two-dimensional image of polarization gradient argument

	Output
	polgrad_arg_dict : dictionary of polarization gradient argument with pixel coordinate keys
	ijpoints         : tuple of pixel coordinates
	'''

	polgrad_arg_dict = {}
	ijpoints         = []

	NAXIS2,NAXIS1 = polgrad_arg.shape

	for j in range(NAXIS2):
		for i in range(NAXIS1):
			arg = polgrad_arg[j,i]
			polgrad_arg_dict[i,j]=arg
			ijpoints.append((i,j))

	return polgrad_arg_dict,ijpoints

def fplotvectors(imagefile,anglefile,deltapix=5,scale=1.,angleunit="deg",coords="wcs",figsize=(20,10)):
	'''
	Plots an image with pseudovectors.
	
	Input
	imagefile : image directory
	anglefile : angle map directory
	deltapix  : the spacing of image pixels to draw pseudovectors
	scale     : a scalefactor for the length of the pseudovectors
	angleunit : the unit of the input angle map (can be deg/degree/degrees or rad/radian/radians)

	Output
	Saves the image in the same directory as imagefile with "_angles.pdf" as the filename extension
	'''

	degree_units   = ["deg","degree","degrees"]
	radian_units   = ["rad","radian","radians"]

	wcs_units      = ["wcs","WCS","world"]
	pixel_units    = ["pix","pixel","pixels"]

	if coords in wcs_units:
		# extract image data and WCS header
		image,header   = fits.getdata(imagefile,header=True)
		NAXIS1,NAXIS2  = header["NAXIS1"],header["NAXIS2"]
		w              = wcs.WCS(header)
	elif coords in pixel_units:
		# extract image data
		image          = fits.getdata(imagefile)
		NAXIS2,NAXIS1  = image.shape
	# extract angle data
	angles         = fits.getdata(anglefile)

	linelist_pix   = []
	linelist_wcs   = []

	for y in range(0,NAXIS2,deltapix):
		# iterate through y pixels
		for x in range(0,NAXIS1,deltapix):
			# iterate through x pixels
			image_xy = image[y,x]
			if np.isnan(image_xy)==False:
				# do not plot angle if image data is NaN
				if angleunit in degree_units:
					# convert angles to radians
					angles_deg = np.copy(angles)
					angles_rad = np.radians(angles)
				elif angleunit in radian_units:
					# convert angles to degrees
					angles_deg = np.degrees(angles)
					angles_rad = np.copy(angles)
				else:
					# raise error
					print "Input angleunit is not defined."
					sys.exit()
				angle_rad = angles_rad[y,x]
				angle_deg = angles_deg[y,x]
				amp       = image[y,x]*100.*scale
				# create line segment in pixel coordinates
				(x1_pix,y1_pix) = (x-amp*np.cos(angle_rad),y-amp*np.sin(angle_rad))
				(x2_pix,y2_pix) = (x+amp*np.cos(angle_rad),y+amp*np.sin(angle_rad))
				line_pix        = np.array([(x1_pix,y1_pix),(x2_pix,y2_pix)])
				if coords in pixel_units:
					linelist_pix.append(line_pix)
				elif coords in wcs_units:
					# create line segment in WCS coordinates (units of degrees)
					x1_wcs,y1_wcs   = w.wcs_pix2world(x1_pix,y1_pix,0)
					x2_wcs,y2_wcs   = w.wcs_pix2world(x2_pix,y2_pix,0)
					line_wcs        = np.array([(x1_wcs,x2_wcs),(y1_wcs,y2_wcs)])
					linelist_wcs.append(line_wcs)

	# plot figure
	if coords in pixel_units:
		# replace NaNs with zeros just for plotting visuals
		image[np.isnan(image)==True] = 0.0

		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111)
		im = ax.imshow(image,vmax=0.05,cmap="Greys_r",origin="lower")
		plt.xlabel("pixels")
		plt.ylabel("pixels")
		lc = LineCollection(linelist_pix,color="red")
		plt.gca().add_collection(lc)
		plt.colorbar(im, ax=ax, orientation="vertical")
		plt.show()
		plt.savefig(imagefile.split(".fits")[0]+"_angles.pdf")

	elif coords in wcs_units:
		fig = plt.figure(figsize=figsize)
		# colorscale
		f = aplpy.FITSFigure(imagefile,figure=fig)
		f.show_grayscale()
		f.set_nan_color("black")
		# pseudovectors
		f.show_lines(linelist_wcs,layer="vectors",color="red")
		# tick coordinates
		f.tick_labels.set_xformat("hh:mm")
		f.tick_labels.set_yformat("dd")
		# axis labels
		f.axis_labels.set_font(size=30)
		# tick labels
		f.tick_labels.show()
		f.tick_labels.set_font(size=30)
		# colour bar
		f.add_colorbar()
		f.colorbar.set_axis_label_text(r"$|\vec{\nabla}\vec{P}|\,\mathrm{(K/arcmin)}$")
		f.colorbar.set_axis_label_font(size=30)
		# scale bar
		f.add_scalebar(1.,color="white",corner="top left")
		f.scalebar.set_label("1 degree")
		f.scalebar.set_font(size=25)
		# remove whitespace
		plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)

		fig.canvas.draw()
		f.save(imagefile.split(".fits")[0]+"_angles.pdf")

def fRMSF(freq_file,weights_file):
	'''
	Calculate the rotation measure spread function (RMSF) and related parameters.

	Input
	freq_file : directory to file containing frequencies in Hertz
	weights_file : directory to file containing weights
	'''

	# arrays
	freq_arr           = np.loadtxt(freq_file,dtype=float) # frequency array in Hertz
	lambda_sq_arr      = (const.c.value/freq_arr)**2.      # lambda^2 array in m^2
	if weights_file==None:
		weights_arr = np.ones(shape=freq_arr.shape)     # weights array
	else:
		weights_arr = np.loadtxt(weights_file,dtype=float) # weights array

	# numbers
	lambda_sq_min      = np.min(lambda_sq_arr)                               # lambda^2 min
	lambda_sq_max      = np.max(lambda_sq_arr)                               # lambda^2 max
	Delta_lambda_sq    = lambda_sq_max-lambda_sq_min                         # lambda^2 range
	delta_lambda_sq    = np.median(np.abs(np.diff(lambda_sq_arr)))           # lambda^2 step size
	Delta_phi          = 3.79/Delta_lambda_sq                                # RM resolution (FWHM)
	phi_max            = 1.9/delta_lambda_sq                                 # maximum RM value
	phi_max_scale      = np.pi/lambda_sq_min                                 # broadest RM feature

	phi_max            = 10.*2.*np.sqrt(3.0) / (lambda_sq_max-lambda_sq_min) # 
	delta_phi          = 0.1*2.*np.sqrt(3.0) / (lambda_sq_max-lambda_sq_min) # 
	delta_freq_Hz      = np.nanmin(np.abs(np.diff(freq_arr)))                #

	# arrays
	phi_array=np.arange(-1.*phi_max,phi_max+1e-6,delta_phi)

	#Output key results to terminal:
	print("RMSF PROPERTIES:")
	#print('Theoretical (unweighted) FWHM:       {:.4g} rad m^-2'.format(2.*np.sqrt(3.) / (lambda_sq_max-lambda_sq_min)))
	#print('Measured FWHM:                       {:.4g} rad m^-2'.format(fwhmRMSFArr))
	print("Resolution in RM:                    {:.4g} rad m^-2".format(Delta_phi))
	print("Broadest RM feature probed:          {:.4g} rad m^-2".format(phi_max_scale))
	print("Maximum RM that can be detected:     {:.4g} rad m^-2".format(phi_max))

def fFDFmom1_1d(FDF_arr,phi_arr,threshold):
	'''
	Computes the first moment of a one-dimensional Faraday dispersion function (FDF).

	Input
	FDF_arr: faraday dispersion function array
	phi_arr : faraday depth array

	Output
	mom1 : first moment of FDF
	'''

	# take absolute value of input FDF
	FDF_arr = np.abs(FDF_arr)
	# apply signal threshold
	_,FDF_arr = fmask_signal(FDF_arr,threshold)

	# compute moment 1
	mom1    = np.nansum(FDF_arr*phi_arr)/np.nansum(FDF_arr)

	return mom1 # rad/m^2

def fFDFmom2_1d(FDF_arr,phi_arr,threshold,sqrt=False):
	'''
	Computes the second moment of a one-dimensional Faraday dispersion function (FDF).
	
	Input
	FDF_arr : faraday dispersion function array
	phi_arr : faraday depth array
	sqrt    : if True, take square root of second moment

	Output
	mom2 : second moment of FDF
	'''

	# take absolute value of input FDF
	FDF_arr = np.abs(FDF_arr)
	# apply signal threshold
	_,FDF_arr = fmask_signal(FDF_arr,threshold)

	# compute moment 1
	mom1 = fFDFmom1_1d(FDF_arr,phi_arr,threshold)

	# compute second moment
	mom2 = np.nansum(FDF_arr*(phi_arr-mom1)**2.) / np.nansum(FDF_arr)
	if sqrt==True:
		mom2 = np.sqrt(mom2)

	return mom2 # (rad/m^2)^2 ; or rad/m^2 if square root is taken

def fFDFmom1_3d(FDF_cube,phi_arr,threshold):
	'''
	Computes the first moment map of a three-dimensional Faraday dispersion function (FDF).

	Input
	FDF_cube : faraday dispersion function cube
	phi_arr  : faraday depth array

	Output
	mom1 : first moment of FDF
	'''

	NAXIS3,NAXIS2,NAXIS1 = FDF_cube.shape
	mom1                 = np.ones(shape=(NAXIS2,NAXIS1))*np.nan

	for ypix in np.arange(NAXIS2):
		for xpix in np.arange(NAXIS1):
			mom1_i          = fFDFmom1_1d(FDF_cube[:,ypix,xpix],phi_arr,threshold)
			mom1[ypix,xpix] = mom1_i

	return mom1 # rad/m^2

def fFDFmom2_3d(FDF_cube,phi_arr,threshold,sqrt=False):
	'''
	Computes the second moment map of a three-dimensional Faraday dispersion function (FDF).

	Input
	FDF_cube : faraday dispersion function cube
	phi_arr  : faraday depth array
	sqrt     : if True, take square root of second moment

	Output
	mom2 : second moment of FDF
	'''

	NAXIS3,NAXIS2,NAXIS1 = FDF_cube.shape
	mom2                 = np.ones(shape=(NAXIS2,NAXIS1))*np.nan

	for ypix in np.arange(NAXIS2):
		for xpix in np.arange(NAXIS1):
			mom2_i          = fFDFmom2_1d(FDF_cube[:,ypix,xpix],phi_arr,threshold,sqrt=sqrt)
			mom2[ypix,xpix] = mom2_i

	return mom2 # (rad/m^2)^2 ; or rad/m^2 if square root is taken













