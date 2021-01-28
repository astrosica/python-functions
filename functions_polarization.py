import sys
import aplpy
import numpy as np
from tqdm import tqdm
import astropy.wcs as wcs
from astropy.io import fits
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy import constants as const
from matplotlib.collections import LineCollection

from functions_misc import fmask_signal
from functions_fits import fheader_3Dto2D

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

def fPI_err(Q,U,Q_err,U_err):
	'''
	Computes the uncertainty in the polarized intensity.

	Input
	Q     : Stokes Q
	U     : Stokes U
	Q_err : uncertainty in Stokes Q
	U_err : uncertainty in Stokes U
	'''

	P = np.sqrt(Q**2.+U**2.)

	Q_term = (Q/P)**2. * Q_err**2.
	U_term = (U/P)**2. * U_err**2.

	P_err = np.sqrt(Q_term + U_term)

	return P_err


def fPI_error(Q,U,QQ,QU,UU):
	'''
	Computes the uncertainty of the the polarized intensity.
	See Equation B.4 in Planck XIX (2015).

	Input
	Q  : Stokes Q
	U  : Stokes U
	QQ : QQ covariance
	QU : QU covariance
	UU : UU covariance

	Output
	PI_error : uncertainty in polarized intensity
	'''

	# compute polarized intensity
	PI = fPI(Q,U)

	PI_error = (Q**2.*QQ + U**2.*UU + 2.*Q*U*QU)/(PI**2.)

	return PI_error

def fpolangle(Q,U,toIAU=False,deg=True):
	'''
	Computes the polarization angle.
	
	Input
	Q     : Stokes Q
	U     : Stokes U
	toIAU : if True, converts from COSMO to IAU convention (default=False)
	deg   : if True, convert angles to degrees for output (default=True)

	Output
	polangle : polarization angle
	'''

	if toIAU==True:
		# if converting from COSMOS to IAU convention
		# flip sign of Stokes U

		# compute polarization angle
		pol_angle = np.mod(0.5*np.arctan2(U*-1.,Q), np.pi)

	elif toIAU==False:
		# don't flip sign of Stokes U

		# compute polarization angle
		pol_angle = np.mod(0.5*np.arctan2(U,Q), np.pi)

	if deg==True:
		# convert from radians to degrees
		pol_angle = np.degrees(pol_angle)

	return pol_angle

def fpolangle_error(Q,U,QQ,QU,UU,deg=True):
	'''
	Computed the uncertainty in the polarization angle.
	See Equation B.3 in Planck XIX (2015).

	Input
	Q   : Stokes Q
	U   : Stokes U
	QQ  : QQ covariance
	QU  : QU covariance
	UU  : UU covariance
	deg : if True, convert angles to degrees for output (default=True)

	Output
	polangle_error : uncertainty in polarization angle
	'''

	num = Q**2.*UU + U**2.*QQ - 2.*Q*U*QU
	den = Q**2.*QQ + U**2.*UU + 2.*Q*U*QU

	# compute uncertainty in polarized intensity
	PI_error = fPI_error(Q,U,QQ,QU,UU)

	if deg==True:
		fac = 28.65
	else:
		fac = np.radians(28.65)

	# compute uncertainty in polarization angle
	polangle_error = fac * np.sqrt(num/den) * PI_error

	return polangle_error

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

	if toIAU==True:
		# if converting from COSMOS to IAU convention flip sign of Stokes U 
		# and flip both Q and U to convert from E to B
		# (i.e., just flip sign of Q)

		# compute B angle
		B_angle = np.mod(0.5*np.arctan2(U,Q*-1.), np.pi)

	elif toIAU==False:
		# don't flip sign of Stokes U
		# but do flip both Q and U to convert from E to B
		# (i.e., flip sign of both Q and U)

		# compute polarization angle
		B_angle = np.mod(0.5*np.arctan2(U*-1.,Q*-1.), np.pi)

	if deg==True:
		# convert from radians to degrees
		B_angle = np.degrees(B_angle)

	return B_angle

def fQU(P,chi,deg=True):
	'''
	Compute Stokes Q and U.

	Input
	P    : polarized intensity
	chi  : polarization angle
	deg : if True, polarization angle is in degrees (otherwise radians) [default=True]

	Output
	Q : Stokes Q
	U : Stokes U
	'''

	# make sure polarization angles are in radians
	if deg==True:
		chi_rad = np.radians(chi)
	elif deg==False:
		chi_rad = chi

	# compute Stokes QU
	Q = P*np.cos(2.*chi_rad)
	U = P*np.sin(2.*chi_rad)

	return Q,U

def fpolanglediff_Stokes(Q_1,U_1,Q_2,U_2,toIAU=False,deg=True):
	'''
	Computes the difference between two polarization angles.
	See Equation 7 in Planck XIX (2015).
	
	Input
	Q_1     : Stokes Q corresponding to reference polarization angle
	U_1     : Stokes U corresponding to reference polarization angle
	Q_2     : Stokes Q corresponding to displaced polarization angle
	U_2     : Stokes U corresponding to displaced polarization angle
	toIAU : if True, converts from COSMO to IAU convention (default=False)
	deg   : if True, convert angles to degrees for output (default=True)

	Output
	pol_angle_diff : difference between polarization angles
	'''

	Q_1 = np.copy(Q_1)
	U_1 = np.copy(U_1)
	Q_2 = np.copy(Q_2)
	U_2 = np.copy(U_2)

	if toIAU==True:
		# if converting from COSMOS to IAU convention
		# flip sign of Stokes U
		U_1 *= -1.
		U_2 *= -1.

	# difference between polarization angles
	pol_angle_diff = 0.5*np.arctan2(Q_2*U_1-Q_1*U_2,Q_2*Q_1-U_2*U_1)

	if deg==True:
		# convert from radians to degrees
		pol_angle_diff = np.degrees(pol_angle)

	return pol_angle_diff

def fpolanglediff(polangle_1,polangle_2,inunits="deg",outunits="deg"):
	'''
	Computes the difference between two polarization angles.
	See Equation 15 in Clark (2019a).
	
	Input
	polangle_1 : 
	polangle_2 :

	Output
	pol_angle_diff : difference between polarization angles
	'''

	degree_units = ["deg","degree","degrees"]
	rad_units    = ["rad","radian","radians"]

	if inunits in degree_units:
		polangle_1_deg = polangle_1.copy()
		polangle_2_deg = polangle_2.copy()
		polangle_1_rad = np.radians(polangle_1)
		polangle_2_rad = np.radians(polangle_2)
	elif inunits in rad_units:
		polangle_1_rad = polangle_1.copy()
		polangle_2_rad = polangle_2.copy()
		polangle_1_deg = np.degrees(polangle_1)
		polangle_2_deg = np.degrees(polangle_2)

	# difference between polarization angles
	num = np.sin(2.*polangle_1_rad)*np.cos(2.*polangle_2_rad) - np.cos(2.*polangle_1_rad)*np.sin(2.*polangle_2_rad)
	den = np.cos(2.*polangle_1_rad)*np.cos(2.*polangle_2_rad) + np.sin(2.*polangle_1_rad)*np.sin(2.*polangle_2_rad)

	pol_angle_diff = 0.5*np.arctan2(num,den)

	if outunits in degree_units:
		pol_angle_diff = np.degrees(pol_angle_diff)

	return pol_angle_diff

def fAM(theta1,theta2,inunits="deg"):
	'''
	Computes the alignment measure (AM) between two angles on the range [0,180).
	See Equation 16 in Clark et al. (2019a).

	Input
	theta1  : angles in units of inunits
	theta2  : angles in units of inunits
	inunits : defines units of input angles

	Output
	AM_full : full array of "AM" measurements
	AM      : alignment measure on the range [-1,1]

	Note: An AM of +1, -1, and 0 implies that the angles are perfectly aligned, perfeclty anti-aligned, and not at all aligned, respectively.
	'''

	# compute angular diference in radians
	deltatheta_rad = fpolanglediff(theta1,theta2,inunits=inunits,outunits="rad")

	# compute AM
	AM_full = np.cos(2.*deltatheta_rad)
	AM      = np.nanmean(AM_full)

	return AM_full,AM

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

def fpolfrac_error(I,Q,U,II,IQ,IU,QQ,QU,UU):
	'''
	Computes the uncertainty in the polarization fraction.

	Input
	I  : Stokes I
	Q  : Stokes Q
	U  : Stokes U
	II : II covariance
	IQ : IQ covariance
	IU : IU covariance
	QQ : QQ covariance
	QU : QU covariance
	UU : UU covariance

	Output
	polfrac_error : uncertainty in polarization fraction
	'''

	# compute polarization fraction
	polfrac = fpolfrac(I,Q,U)

	# compute uncertainty in polarization fraction
	a_den = (polfrac**2.) * (I**4.)
	a     = 1./fac_den
	b     = Q**2.*QQ + U**2.*UU + (II/(I**2.))*(Q**2.+U**2.)**2. + 2.*Q*U*QU - (2.*U*(Q**2.+U**2.)*IQ)/I - (2.*U*(Q**2.+U**2.)*IU)/I

	polfrac_error = np.sqrt(a*b)

	return polfrac_error

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

def fpolgradnorm_crossterms(Q,U):
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

	# compute the polarized intensity
	P = np.sqrt(Q**2.+U**2.)
	
	# compute normalized polarization gradient
	polgrad_norm = polgrad/P
	
	return polgrad_norm

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
	Computes the tangential component of the polarization gradient.
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

def fgradchi(Q,U):
	'''
	Computes the angular version of the polarization gradient.
	See Equation 6 in Planck XII (2018).

	Input
	Q : Stokes Q
	U : Stokes U

	Output
	gradchi : 
	'''

	# compute polarized intensity
	P = fPI(Q,U)

	# compute main terms in gradphi
	QP = Q/P
	UP = U/P

	# compute Stokes spatial gradients
	QP_grad_y,QP_grad_x = np.gradient(QP)
	UP_grad_y,UP_grad_x = np.gradient(UP)

	# compute gradient of angular component
	gradchi = np.sqrt((QP_grad_x)**2. + (QP_grad_y)**2. + (UP_grad_x)**2. + (UP_grad_y)**2.)

	return gradchi

def fSest(Q,U,delta):
	'''
	Computes an estimate of the polarization angle dispersion function.
	See Equation 7 in Planck XII (2018).

	Input
	Q     : Stokes Q
	U     : Stokes U
	delta : lag in pixels
	'''

	# compute gradient of polarization angle
	gradchi = fgradchi(Q,U)

	# compute Sest
	Sest = delta*gradchi/(2.*np.sqrt(2.))

	return Sest


def fderotate(pangle,RM,freq,inunit,outunit):
	'''
	Computes the de-rotated polarization angles.

	Input
	pangle  : polarization angle                 [degrees or radians]
	RM      : rotation measure                   [rad/m^2]
	freq    : frequency channels                 [Hz]
	inunit  : units of input polarization angle  [degrees or radians]
	outunit : units of output polarization angle [degrees or radians]
	'''

	# compute weighted average of wavelength squared
	wavel_sq   = fwavel(freq)**2.
	weights    = np.ones(shape=wavel_sq.shape)
	K          = 1.0/np.sum(weights)
	wavel_0_sq = K*np.sum(weights*wavel_sq)

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

	pangle_0_rad = np.mod(pangle_rad-RM*wavel_0_sq,np.pi)
	pangle_0_deg = np.degrees(pangle_0_rad)

	if outunit in degree_units:
		# if output polarization angle is in degrees
		pangle_0 = pangle_0_deg
	elif outunit in rad_units:
		# if output polarization angle is in radians
		pangle_0 = pangle_0_rad

	return pangle_0

def fderotate_err(RMSF_FWHM,SNR,lambda_sq,deg=True):
	'''
	Commputes the uncertainty in de-rotated polarization angles.

	Input
	RMSF_FWHM : the FWHM of the RMSF                     [rad/m^2]
	SNR       : signal-to-noise                          [dimensionless]
	lambda_sq : wavelength^2                             [m^2]
	deg       : if true, converts uncertainty to degrees [default=True]


	Output
	derotate_err : unceertainty in de-rotated angles [deg]
	'''

	derotate_err = (RMSF_FWHM/(2.*SNR))*lambda_sq

	if deg==True:
		derotate_err = np.degrees(derotate_err)

	return derotate_err

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
					sys.exit() # pol_angle = np.mod(0.5*np.arctan2(U,Q), np.pi)
				angle_rad = angles_rad[y,x]
				angle_deg = angles_deg[y,x]
				amp       = image[y,x]*100.*scale
				# create line segment in pixel coordinates
				(x1_pix,y1_pix) = (x-amp*np.sin(angle_rad),y+amp*np.cos(angle_rad))
				(x2_pix,y2_pix) = (x+amp*np.sin(angle_rad),y-amp*np.cos(angle_rad))
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
		#f.show_grayscale()
		f.show_colorscale(cmap="hot",vmin=0.0,vmax=0.03)
		f.set_nan_color("black")
		# pseudovectors
		f.show_lines(linelist_wcs,layer="vectors",color="white",linewidth=0.5)
		# tick coordinates
		f.tick_labels.set_xformat("hh:mm")
		f.tick_labels.set_yformat("dd")
		# axis labels
		f.axis_labels.set_font(size=20)
		# tick labels
		f.tick_labels.show()
		f.tick_labels.set_font(size=20)
		# colour bar
		f.add_colorbar()
		f.colorbar.set_axis_label_text(r"$|\vec{\nabla}\vec{P}|_\mathrm{max}\,\mathrm{(K/arcmin)}$")
		#f.colorbar.set_axis_label_text(r"$I_{857}\,\mathrm{(MJy/sr)}$")
		f.colorbar.set_axis_label_font(size=20)
		# scale bar
		f.add_scalebar(5.,color="white",corner="top left")
		f.scalebar.set_label("5 degrees")
		f.scalebar.set_font(size=20)
		# remove whitespace
		plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)

		fig.canvas.draw()
		f.save(imagefile.split(".fits")[0]+"_angles.pdf")

def frun_RM1d_iterate(do_RMsynth_1D_dir,Q_filedir,U_filedir,Q_err,U_err,freq_Hz_filedir,rmsynth_inputfiledir,options):
	'''
	Iteratively runs 1D RM synthesis on a cube of data.
	
	Input
	do_RMsynth_1D_dir     : 
	Q_filedir             : 
	U_filedir             : 
	Q_err                 : 
	U_err                 : 
	freq_Hz_filedir       : 
	rmsynth_inputfiledir  : 
	options               : 
	
	Output
	
	'''

	rmsynth_outputfiledir = rmsynth_inputfiledir.split(".")[0]+"_RMsynth."+rmsynth_inputfiledir.split(".")[-1]

	# extract data
	Q_data       = fits.getdata(Q_filedir)
	U_data       = fits.getdata(U_filedir)
	freq_Hz_data = np.loadtxt(freq_Hz_filedir)
	Q_header_2D  = fheader_3Dto2D(Q_filedir,Q_filedir,write=False)

	# turn uncertainties into arrays
	sigma_Q = np.ones(shape=Q_data.shape[0])*Q_err
	sigma_U = np.ones(shape=U_data.shape[0])*U_err

	# create nan arrays to later replace with RM synthesis results
	dFDFcorMAD           = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dFDFrms              = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	phiPeakPIchan_rm2    = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dPhiPeakPIchan_rm2   = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	ampPeakPIchan        = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	ampPeakPIchanEff     = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dAmpPeakPIchan       = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	snrPIchan            = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	indxPeakPIchan       = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	peakFDFimagChan      = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	peakFDFrealChan      = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	polAngleChan_deg     = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dPolAngleChan_deg    = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	polAngle0Chan_deg    = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dPolAngle0Chan_deg   = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	phiPeakPIfit_rm2     = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dPhiPeakPIfit_rm2    = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	ampPeakPIfit         = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	ampPeakPIfitEff      = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dAmpPeakPIfit        = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	snrPIfit             = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	indxPeakPIfit        = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	peakFDFimagFit       = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	peakFDFrealFit       = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	polAngleFit_deg      = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dPolAngleFit_deg     = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	polAngle0Fit_deg     = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dPolAngle0Fit_deg    = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	Ifreq0               = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	#polyCoeffs           = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	IfitStat             = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	IfitChiSqRed         = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	lam0Sq_m2            = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	freq0_Hz             = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	fwhmRMSF             = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dQU                  = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dFDFth               = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	#units                = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	min_freq             = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	max_freq             = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	N_channels           = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	median_channel_width = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	fracPol              = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	sigmaAddQ            = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dSigmaAddMinusQ      = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dSigmaAddPlusQ       = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	sigmaAddU            = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dSigmaAddMinusU      = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan
	dSigmaAddPlusU       = np.ones(shape=(Q_data.shape[1],Q_data.shape[2]))*np.nan

	# iterate over spatial y-pixel coordinates
	for _y in tqdm(range(Q_data.shape[1])):
		# iterate over spatial x-pixel coordinates
		for _x in range(Q_data.shape[2]):
			# extract Stokes data at spatial pixel
			Q_xy  = Q_data[:,_y,_x]
			U_xy  = U_data[:,_y,_x]
			# write file containing input for RM synthesis 1D
			np.savetxt(rmsynth_inputfiledir,np.transpose([freq_Hz_data,Q_xy,U_xy,sigma_Q,sigma_U]))
			# run RM synthesis
			#os.system(do_RMsynth_1D_dir+" "+str(rmsynth_inputfiledir)+" "+options)
			DEVNULL = open(os.devnull, 'wb')
			subprocess.call(do_RMsynth_1D_dir+" "+str(rmsynth_inputfiledir)+" "+options,stderr=DEVNULL,shell=True)
			# extract RM synthesis output
			with open(rmsynth_outputfiledir) as f:
				lines = f.readlines()
				dFDFcorMAD_xy           = float(lines[0].split("=")[1])
				dFDFrms_xy              = float(lines[1].split("=")[1])
				phiPeakPIchan_rm2_xy    = float(lines[2].split("=")[1])
				dPhiPeakPIchan_rm2_xy   = float(lines[3].split("=")[1])
				ampPeakPIchan_xy        = float(lines[4].split("=")[1])
				ampPeakPIchanEff_xy     = float(lines[5].split("=")[1])
				dAmpPeakPIchan_xy       = float(lines[6].split("=")[1])
				snrPIchan_xy            = float(lines[7].split("=")[1])
				indxPeakPIchan_xy       = float(lines[8].split("=")[1])
				peakFDFimagChan_xy      = float(lines[9].split("=")[1])
				peakFDFrealChan_xy      = float(lines[10].split("=")[1])
				polAngleChan_deg_xy     = float(lines[11].split("=")[1])
				dPolAngleChan_deg_xy    = float(lines[12].split("=")[1])
				polAngle0Chan_deg_xy    = float(lines[13].split("=")[1])
				dPolAngle0Chan_deg_xy   = float(lines[14].split("=")[1])
				phiPeakPIfit_rm2_xy     = float(lines[15].split("=")[1])
				dPhiPeakPIfit_rm2_xy    = float(lines[16].split("=")[1])
				ampPeakPIfit_xy         = float(lines[17].split("=")[1])
				ampPeakPIfitEff_xy      = float(lines[18].split("=")[1])
				dAmpPeakPIfit_xy        = float(lines[19].split("=")[1])
				snrPIfit_xy             = float(lines[20].split("=")[1])
				indxPeakPIfit_xy        = float(lines[21].split("=")[1])
				peakFDFimagFit_xy       = float(lines[22].split("=")[1])
				peakFDFrealFit_xy       = float(lines[23].split("=")[1])
				polAngleFit_deg_xy      = float(lines[24].split("=")[1])
				dPolAngleFit_deg_xy     = float(lines[25].split("=")[1])
				polAngle0Fit_deg_xy     = float(lines[26].split("=")[1])
				dPolAngle0Fit_deg_xy    = float(lines[27].split("=")[1])
				Ifreq0_xy               = float(lines[28].split("=")[1])
				#polyCoeffs_xy           = str(lines[29].split("=")[1])
				IfitStat_xy             = float(lines[30].split("=")[1])
				IfitChiSqRed_xy         = float(lines[31].split("=")[1])
				lam0Sq_m2_xy            = float(lines[32].split("=")[1])
				freq0_Hz_xy             = float(lines[33].split("=")[1])
				fwhmRMSF_xy             = float(lines[34].split("=")[1])
				dQU_xy                  = float(lines[35].split("=")[1])
				dFDFth_xy               = float(lines[36].split("=")[1])
				#units_xy                = str(lines[37].split("=")[1])
				min_freq_xy             = float(lines[38].split("=")[1])
				max_freq_xy             = float(lines[39].split("=")[1])
				N_channels_xy           = float(lines[40].split("=")[1])
				median_channel_width_xy = float(lines[41].split("=")[1])
				fracPol_xy              = float(lines[42].split("=")[1])
				sigmaAddQ_xy            = float(lines[43].split("=")[1])
				dSigmaAddMinusQ_xy      = float(lines[44].split("=")[1])
				dSigmaAddPlusQ_xy       = float(lines[45].split("=")[1])
				sigmaAddU_xy            = float(lines[46].split("=")[1])
				dSigmaAddMinusU_xy      = float(lines[47].split("=")[1])
				dSigmaAddPlusU_xy       = float(lines[48].split("=")[1])
			# fill in arrays with results
			dFDFcorMAD[_y,_x]           = dFDFcorMAD_xy
			dFDFrms[_y,_x]              = dFDFrms_xy
			phiPeakPIchan_rm2[_y,_x]    = phiPeakPIchan_rm2_xy
			dPhiPeakPIchan_rm2[_y,_x]   = dPhiPeakPIchan_rm2_xy
			ampPeakPIchan[_y,_x]        = ampPeakPIchan_xy
			ampPeakPIchanEff[_y,_x]     = ampPeakPIchanEff_xy
			dAmpPeakPIchan[_y,_x]       = dAmpPeakPIchan_xy
			snrPIchan[_y,_x]            = snrPIchan_xy
			indxPeakPIchan[_y,_x]       = indxPeakPIchan_xy
			peakFDFimagChan[_y,_x]      = peakFDFimagChan_xy
			peakFDFrealChan[_y,_x]      = peakFDFrealChan_xy
			polAngleChan_deg[_y,_x]     = polAngleChan_deg_xy
			dPolAngleChan_deg[_y,_x]    = dPolAngleChan_deg_xy
			polAngle0Chan_deg[_y,_x]    = polAngle0Chan_deg_xy
			dPolAngle0Chan_deg[_y,_x]   = dPolAngle0Chan_deg_xy
			phiPeakPIfit_rm2[_y,_x]     = phiPeakPIfit_rm2_xy
			dPhiPeakPIfit_rm2[_y,_x]    = dPhiPeakPIfit_rm2_xy
			ampPeakPIfit[_y,_x]         = ampPeakPIfit_xy
			ampPeakPIfitEff[_y,_x]      = ampPeakPIfitEff_xy
			dAmpPeakPIfit[_y,_x]        = dAmpPeakPIfit_xy
			snrPIfit[_y,_x]             = snrPIfit_xy
			indxPeakPIfit[_y,_x]        = indxPeakPIfit_xy
			peakFDFimagFit[_y,_x]       = peakFDFimagFit_xy
			peakFDFrealFit[_y,_x]       = peakFDFrealFit_xy
			polAngleFit_deg[_y,_x]      = polAngleFit_deg_xy
			dPolAngleFit_deg[_y,_x]     = dPolAngleFit_deg_xy
			polAngle0Fit_deg[_y,_x]     = polAngle0Fit_deg_xy
			dPolAngle0Fit_deg[_y,_x]    = dPolAngle0Fit_deg_xy
			Ifreq0[_y,_x]               = Ifreq0_xy
			#polyCoeffs[_y,_x]           = polyCoeffs_xy
			IfitStat[_y,_x]             = IfitStat_xy
			IfitChiSqRed[_y,_x]         = IfitChiSqRed_xy
			lam0Sq_m2[_y,_x]            = lam0Sq_m2_xy
			freq0_Hz[_y,_x]             = freq0_Hz_xy
			fwhmRMSF[_y,_x]             = fwhmRMSF_xy
			dQU[_y,_x]                  = dQU_xy
			dFDFth[_y,_x]               = dFDFth_xy
			#units[_y,_x]                = units_xy
			min_freq[_y,_x]             = min_freq_xy
			max_freq[_y,_x]             = max_freq_xy
			N_channels[_y,_x]           = N_channels_xy
			median_channel_width[_y,_x] = median_channel_width_xy
			fracPol[_y,_x]              = fracPol_xy
			sigmaAddQ[_y,_x]            = sigmaAddQ_xy
			dSigmaAddMinusQ[_y,_x]      = dSigmaAddMinusQ_xy
			dSigmaAddPlusQ[_y,_x]       = dSigmaAddPlusQ_xy
			sigmaAddU[_y,_x]            = sigmaAddU_xy
			dSigmaAddMinusU[_y,_x]      = dSigmaAddMinusU_xy
			dSigmaAddPlusU[_y,_x]       = dSigmaAddPlusU_xy
'''
		fits.writeto(rmsynth_outputfiledir+"dFDFcorMAD.fits",dFDFcorMAD,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dFDFrms.fits",dFDFrms,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"phiPeakPIchan_rm2.fits",phiPeakPIchan_rm2,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dPhiPeakPIchan_rm2.fits",dPhiPeakPIchan_rm2,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"ampPeakPIchan.fits",ampPeakPIchan,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"ampPeakPIchanEff.fits",ampPeakPIchanEff,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dAmpPeakPIchan.fits",dAmpPeakPIchan,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"snrPIchan.fits",snrPIchan,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"indxPeakPIchan.fits",indxPeakPIchan,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"peakFDFimagChan.fits",peakFDFimagChan,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"peakFDFrealChan.fits",peakFDFrealChan,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"polAngleChan_deg.fits",polAngleChan_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dPolAngleChan_deg.fits",dPolAngleChan_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"polAngle0Chan_deg.fits",polAngle0Chan_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dPolAngle0Chan_deg.fits",dPolAngle0Chan_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"phiPeakPIfit_rm2.fits",phiPeakPIfit_rm2,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dPhiPeakPIfit_rm2.fits",dPhiPeakPIfit_rm2,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"ampPeakPIfit.fits",ampPeakPIfit,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"ampPeakPIfitEff.fits",ampPeakPIfitEff,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dAmpPeakPIfit.fits",dAmpPeakPIfit,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"snrPIfit.fits",snrPIfit,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"indxPeakPIfit.fits",indxPeakPIfit,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"peakFDFimagFit.fits",peakFDFimagFit,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"peakFDFrealFit.fits",peakFDFrealFit,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"polAngleFit_deg.fits",polAngleFit_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dPolAngleFit_deg.fits",dPolAngleFit_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"polAngle0Fit_deg.fits",polAngle0Fit_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dPolAngle0Fit_deg.fits",dPolAngle0Fit_deg,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"Ifreq0.fits",Ifreq0,Q_header_2D,overwrite=True)
		#fits.writeto(rmsynth_outputfiledir+"polyCoeffs.fits",polyCoeffs,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"IfitStat.fits",IfitStat,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"IfitChiSqRed.fits",IfitChiSqRed,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"lam0Sq_m2.fits",lam0Sq_m2,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"freq0_Hz.fits",freq0_Hz,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"fwhmRMSF.fits",fwhmRMSF,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dQU.fits",dQU,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dFDFth.fits",dFDFth,Q_header_2D,overwrite=True)
		#fits.writeto(rmsynth_outputfiledir+"units.fits",units,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"min_freq.fits",min_freq,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"max_freq.fits",max_freq,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"N_channels.fits",N_channels,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"median_channel_width.fits",median_channel_width,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"fracPol.fits",fracPol,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"sigmaAddQ.fits",sigmaAddQ,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dSigmaAddMinusQ.fits",dSigmaAddMinusQ,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dSigmaAddPlusQ.fits",dSigmaAddPlusQ,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"sigmaAddU.fits",sigmaAddU,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dSigmaAddMinusU.fits",dSigmaAddMinusU,Q_header_2D,overwrite=True)
		fits.writeto(rmsynth_outputfiledir+"dSigmaAddPlusU.fits",dSigmaAddPlusU,Q_header_2D,overwrite=True)
'''

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

def fFDFmom0_1d(FDF_arr,phi_arr,threshold):
	'''
	Computes the zeroth moment of a one-dimensional Faraday dispersion function (FDF).

	Input
	FDF_arr : faraday dispersion function array
	phi_arr : faraday depth array

	Output
	mom0 : zeroth moment of FDF
	'''

	# apply signal threshold
	_,FDF_arr = fmask_signal(FDF_arr,threshold)

	# take sum
	mom0 = np.nansum(FDF_arr)

	return mom0

def fFDFmom1_1d(FDF_arr,phi_arr,threshold):
	'''
	Computes the first moment of a one-dimensional Faraday dispersion function (FDF).

	Input
	FDF_arr : faraday dispersion function array
	phi_arr : faraday depth array

	Output
	mom1 : first moment of FDF
	'''

	# apply signal threshold
	_,FDF_arr = fmask_signal(FDF_arr,threshold)

	# compute moment 0
	mom0 = fFDFmom0_1d(FDF_arr,phi_arr,threshold)

	# compute moment 1
	mom1    = np.nansum(FDF_arr*phi_arr,axis=0)/mom0

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
	
	# apply signal threshold
	_,FDF_arr = fmask_signal(FDF_arr,threshold)

	# compute moment 0
	mom0 = fFDFmom0_1d(FDF_arr,phi_arr,threshold)

	# compute moment 1
	mom1 = fFDFmom1_1d(FDF_arr,phi_arr,threshold)

	# compute second moment
	mom2 = np.nansum(FDF_arr*(phi_arr-mom1)**2.) / mom0
	if sqrt==True:
		mom2 = np.sqrt(mom2)

	return mom2 # (rad/m^2)^2 ; or rad/m^2 if square root is taken

def fFDFmom0_3d(FDF_arr,phi_arr,threshold):
	'''
	Computes the zeroth moment of a three-dimensional Faraday dispersion function (FDF).

	Input
	FDF_arr : faraday dispersion function array
	phi_arr : faraday depth array

	Output
	mom0 : zeroth moment of FDF
	'''

	# apply signal threshold
	_,FDF_arr = fmask_signal(FDF_arr,threshold)

	# take sum
	mom0 = np.nansum(FDF_arr,axis=0)

	return mom0

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