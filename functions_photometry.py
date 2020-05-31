import numpy as np

def fAJHK(Hmag,Hmag_err,G2mag,G2mag_err):
    '''
    Computes the J-, H-, and K-band photometric extinctions using the RJCE techinque.
    See equation 1 in Majewski et al. (2011) and Table 1 in Indebetouw et al. (2005)

    Input
    Hmag      : H-band magnitude
    Hmag_err  : error in H-band magnitude
    G2mag     : GLIMPSE channel-2 magnitude
    G2mag_err : error in GLIMPSE channel-2 magnitude

    Output
    AJ    : J-band extinction
    AJerr : error in J-band extinction
    AH    : H-band extinction
    AHerr : error in H-band extinction
    AK    : K-band extinction
    AKerr : error in K-band extinction
    '''

    # replace None's with infty's so we don't have to deal with missing numbers
    if (isinstance(Hmag,float)==True) or (isinstance(Hmag,int)==True):
        if Hmag is None:
            Hmag = np.infty
        if Hmag_err is None:
            Hmag_err = np.infty
        if G2mag is None:
            G2mag = np.infty
        if G2mag_err is None:
            G2mag_err = np.infty
    elif (isinstance(Hmag,np.ndarray)==True) or (isinstance(Hmag,list)==True):
        if Hmag is None:
            Hmag = np.ones(shape=Hmag) * np.infty
        if Hmag_err is None:
            Hmag_err = np.ones(shape=Hmag) * np.infty
        if G2mag is None:
            G2mag = np.ones(shape=Hmag) * np.infty
        if G2mag_err is None:
            G2mag_err = np.ones(shape=Hmag) * np.infty

    # RJCE techinque
    AK    = 0.918 * (Hmag - G2mag - 0.08)
    AKerr = 0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)

    # scaling relations from Majewski
    AJ    = 2.5 * AK
    AJerr = 2.5 * AKerr

    # scaling relations from Majewski
    AH    = 1.55 * AK
    AHerr = 1.55 * AKerr

    return AJ, AJerr, AH, AHerr, AK, AKerr

def fAJHK_0(Jmag,Jmag_err,Hmag,Hmag_err,Kmag,Kmag_err,G2mag,G2mag_err):
    '''
    Computes the J-, H-, and K-band intrinsic photometric magnitudes using the RJCE techinque.
    See equation 1 in Majewski et al. (2011) and Table 1 in Indebetouw et al. (2005)

    Input
    Jmag      : J-band magnitude (optional)
    Jmag_err  : error in J-band magnitude (optional)
    Hmag      : H-band magnitude (required by the RJCE techinque)
    Hmag_err  : error in H-band magnitude (required by the RJCE techinque)
    Kmag      : K-band magnitude (optional)
    Kmag_err  : error in K-band magnitude (optional)
    G2mag     : GLIMPSE channel-2 magnitude (required by the RJCE techinque)
    G2mag_err : error in GLIMPSE channel-2 magnitude (required by the RJCE techinque)

    Output
    J_0    : J-band instinsic magnnintude
    J_0err : error in J-band instinsic magnnintude
    H_0    : H-band instinsic magnnintude
    H_0err : error in H-band instinsic magnnintude
    K_0    : K-band instinsic magnnintude
    K_0err : error in K-band instinsic magnnintude
    '''

    # replace None's with infty's so we don't have to deal with missing numbers
    if (isinstance(Hmag,float)==True) or (isinstance(Hmag,int)==True):
        if Jmag is None:
            Jmag = np.infty
        if Jmag_err is None:
            Jmag_err = np.infty
        if Hmag is None:
            Hmag = np.infty
        if Hmag_err is None:
            Hmag_err = np.infty
        if Kmag is None:
            Kmag = np.infty
        if Kmag_err is None:
            Kmag_err = np.infty
        if G2mag is None:
            G2mag = np.infty
        if G2mag_err is None:
            G2mag_err = np.infty
    elif (isinstance(Hmag,np.ndarray)==True) or (isinstance(Hmag,list)==True):
        if Jmag is None:
            Jmag     = np.ones(shape=Hmag) * np.infty
        if Jmag_err is None:
            Jmag_err = np.ones(shape=Hmag) * np.infty
        if Hmag is None:
            Hmag     = np.ones(shape=Hmag) * np.infty
        if Hmag_err is None:
            Hmag_err = np.ones(shape=Hmag) * np.infty
        if Kmag is None:
            Kmag     = np.ones(shape=Hmag) * np.infty
        if Kmag_err is None:
            Kmag_err = np.ones(shape=Hmag) * np.infty

    # RJCE techinque (required)
    AK    = 0.918 * (Hmag - G2mag - 0.08)
    AH    = 1.55 * AK
    AJ    = 2.5 * AK

    # uncertainty in AK, AJ, AH
    AKerr = 0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)
    AJerr = 2.5 * AKerr
    AHerr = 1.55*0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)

    # intrinsic K-, J-, and H-band magnitudes
    H0 = Hmag - AH
    J0 = Jmag - AJ
    K0 = Kmag - AK

    # uncertainty in intrinsic K-, J-, and H-band magnitudes
    H0_err = np.sqrt((1.-1.55*0.918)**2.*Hmag_err**2. + (1.55*0.918)**2.*G2mag_err**2.)
    J0_err = np.sqrt(Jmag_err**2. + AJerr**2.)
    K0_err = np.sqrt(Kmag_err**2. + AKerr**2.)

    return J0,J0_err,H0,H0_err,K0,K0_err

def fLJHK(J0,H0,K0,p_mas):
    '''
    Computes the distance-corrected intrinsic J-, H-, and K-band luminosities usinng Gaia parallaxes,

    Input:
    J0    : extinction-corrected J-band magintude
    H0    : extinction-corrected H-band magintude
    K0    : extinction-corrected K-band magintude
    p_mas : Gaia parallaxes in milli-arcsec
    '''

    # replace None's with infty's so we don't have to deal with missing numbers
    if (isinstance(p_mas,float)==True) or (isinstance(p_mas,int)==True):
        if J0 is None:
            J0 = np.infty
        if H0 is None:
            H0 = np.infty
        if K0 is None:
            K0 = np.infty
    elif (isinstance(p_mas,np.ndarray)==True) or (isinstance(p_mas,list)==True):
        if J0 is None:
            J0 = np.ones(shape=p_mas.shape) * np.infty
        if H0 is None:
            H0 = np.ones(shape=p_mas.shape) * np.infty
        if K0 is None:
            K0 = np.ones(shape=p_mas.shape) * np.infty

    # Gaia parallaxes --> distances
    p_arcsec,d_pc,d_10pc,d_Mpc = fGaiaParallax(p_mas)

    # correct intrinsic magnintudes for Gaia distances
    L_J = J0 - 5.*np.log10(d_10pc)
    L_H = H0 - 5.*np.log10(d_10pc)
    L_K = K0 - 5.*np.log10(d_10pc)

    return L_J,L_H,L_K

def fGaiaParallax(p_mas):
    '''
    Computes distance using Gaia parallax.

    Input
    p_mas : Gaia parallax in milli-arcseconds

    Output
    p_as   : parallax
    d_pc   : 
    d_10pc : 
    '''

    # convert parallax from milli-arcsec to arcsec
    p_arcsec = p_mas*1E-3

    # compute distances
    d_pc     = 1./p_arcsec
    d_10pc   = d_pc/10.
    d_Mpc    = d_pc/1E6

    return p_arcsec,d_pc,d_10pc,d_Mpc

def fmagerrtosnr(magerr):
	'''
	Converts photometric magnitude uncertainty to signal-to-noise ratio.
	'''
	magsnr = 1./magerr
	return magsnr

def fmagsnrtoerr(magsnr):
	'''
	Converts photometric magnitude signal-to-noise ratio to uncertainty.
	'''
	magerr = 1./abs(magsnr)
	return magerr

def fconvert_AB_vega(mag_AB,zeropoint):
	'''
	Converts AB magnitudes to the Vega magnitude scale.
	'''
	mag_Vega = mag_AB - zeropoint
	return mag_Vega

def fcolour_err(mag1,mag1err,mag2,mag2err):
	'''
	Computes photometric colour and its uncertainty.
	'''
	
	colour = mag1 - mag2
	colour_err = np.sqrt(mag1err**2. + mag2err**2.)
	
	return colour,colour_err