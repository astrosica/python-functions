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