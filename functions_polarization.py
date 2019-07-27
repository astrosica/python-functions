import numpy as np
from scipy import interpolate
from functions_misc import fmaptheta_halfpolar_to_halfpolar

def fPI(Q,U):
	'''
	Constructs the polarized intensity given Stokes Q and U maps.
	
	Q : Stokes Q data
	U : Stokes U data
	'''
	
	# compute polarized intensity
	PI = np.sqrt(Q**2.+U**2.)
	
	return PI

def fPI_debiased(Q,U,Q_std,U_std):
	'''
	Constructs the de-biased polarized intensity given Stokes Q and U maps along with estimates of their noise.
	
	Q     : Stokes Q data
	U     : Stokes U data
	Q_std : Stokes Q noise standard deviation
	U_std : Stokes U noise standard deviation
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

def fpolgradnorm(Q,U):
	'''
	Constructs the normalized spatial polarization gradient given Stokes Q and U maps.
	
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
	
	# compute the polarized intensity
	P = np.sqrt(Q**2.+U**2.)
	
	# compute normalized polarization gradient
	polgrad_norm = polgrad/P
	
	return polgrad_norm

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
	
	# compute polarization gradient
	polgrad = np.sqrt(0.5*a + 0.5*np.sqrt(b) )
	
	return polgrad

def fpolgradarg(Q,U,fromnorth=True,parallel=True,deg=True):
	'''
	Computes the argument of the maximum complete spatial polarization gradient with the cross-terms included given Stokes Q and U maps.
	
	Q        : Stokes Q data
	U        : Stokes U data
	deg      : if True, converts to degrees at the end
	parallel : if True, compute angle parallel (rather then perpendicular) to polarization gradient structures
	'''
	
	# compute Stokes spatial gradients
	Q_grad   = np.gradient(Q)
	U_grad   = np.gradient(U)
	
	# define components of spatial gradients
	Q_grad_x = Q_grad[0]
	Q_grad_y = Q_grad[1]
	U_grad_x = U_grad[0]
	U_grad_y = U_grad[1]

	# compute argument of polarization gradient
	a = np.sign(Q_grad_x*Q_grad_y + U_grad_x*U_grad_y)
	b = np.sqrt(Q_grad_y**2.+U_grad_y**2.)
	c = np.sqrt(Q_grad_x**2.+U_grad_x**2.)

	polgrad_arg = np.arctan(a*b/c)

	if fromnorth==True:
		# compute argument from North (as the RHT does)
		polgrad_arg = fmaptheta_halfpolar_to_halfpolar(polgrad_arg)

	if parallel==True:
		# compute argument parallel (rather than perpendicular) to polarization gradients
		polgrad_arg_lower_cond                  = np.where(polgrad_arg<np.pi/2.)
		polgrad_arg_upper_cond                  = np.where(polgrad_arg>=np.pi/2.)
		polgrad_arg_lower_x,polgrad_arg_lower_y = polgrad_arg_lower_cond[0],polgrad_arg_lower_cond[1]
		polgrad_arg_upper_x,polgrad_arg_upper_y = polgrad_arg_upper_cond[0],polgrad_arg_upper_cond[1]
		polgrad_arg_lower_xy                    = np.array(zip(polgrad_arg_lower_x,polgrad_arg_lower_y))
		polgrad_arg_upper_xy                    = np.array(zip(polgrad_arg_upper_x,polgrad_arg_upper_y))
		# iterate through each pixel and compute orthogonal angle
		for xy in polgrad_arg_lower_xy:
			x,y = xy[0],xy[1]
			polgrad_arg[x,y]+=np.pi/2.
		for xy in polgrad_arg_upper_xy:
			x,y = xy[0],xy[1]
			polgrad_arg[x,y]-=np.pi/2.

	# convert to degrees
	polgrad_arg = np.degrees(polgrad_arg)

	return polgrad_arg

def fmaskpolgradarg(angles,min,max,interp=False):
	'''
	Masks the argument of polarization gradient.

	Inputs
	angles : 
	min    : 
	max    : 
	interp : boolean to determine if masked pixels will be interpolated over (default=False)
	'''

	mask = np.ones(shape=angles.shape) # initialize mask

	for i in range(len(min)):
		min_i             = min[i]
		max_i             = max[i]
		mask_angles       = np.where((angles>=min_i) & (angles<=max_i))
		mask[mask_angles] = np.nan

	# mask angles
	angles_masked = angles*mask

	if interp==True:
		# create pixel grid
		x      = np.arange(0, angles.shape[1])
		y      = np.arange(0, angles.shape[0])
		xx, yy = np.meshgrid(x, y)
		# collect valid values
		nanmask      = np.ma.masked_invalid(mask).mask
		x1_valid     = xx[~nanmask]
		y1_valid     = yy[~nanmask]
		angles_valid = angles_masked[~nanmask]
		# interpolate
		angles_masked_interp = interpolate.griddata((x1_valid, y1_valid), angles_valid.ravel(),(xx, yy),method="cubic")

		return mask,angles_masked_interp

	else:
		return mask,angles_masked

def fpolgradnorm_crossterms(Q,U):
	'''
	Constructs the complete normalized spatial polarization gradient with the cross-terms included given Stokes Q and U maps.
	
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
	polgrad = np.sqrt(0.5*a + 0.5*np.sqrt(b) )
	
	# compute the polarized intensity
	P = np.sqrt(Q**2.+U**2.)
	
	# compute the normalized polarization gradient
	polgrad_norm = polgrad/P
	
	return polgrad_norm

def fpolgrad_rad(Q,U):
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
	
	# compute radial component of polarization gradient
	polgrad_rad = np.sqrt(polgrad_rad_num/polgrad_rad_den)
	
	return polgrad_rad
    
def fpolgrad_tan(Q,U):
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
	
	polgrad_tan_num = (Q*U_grad_x+U*Q_grad_x)**2. + (Q*U_grad_y-U*Q_grad_y)**2.
	polgrad_tan_den = Q**2.+U**2.
	
	# compute tangential component of polarization gradient
	polgrad_tan = np.sqrt(polgrad_tan_num/polgrad_tan_den)
	
	return polgrad_tan
