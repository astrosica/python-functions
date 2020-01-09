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

    # RJCE techinque
    AK    = 0.918 * (Hmag - G2mag - 0.08)
    if Hmag_err!=None and G2mag_err!=None:
        AKerr = 0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)

    # scaling relations from Majewski
    AJ    = 2.5 * AK
    if AKerr!=None:
        AJerr = 2.5 * AKerr

    # scaling relations from Majewski
    AH    = 1.55 * AK
    if AKerr!=None:
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

    # RJCE techinque (required)
    AK    = 0.918 * (Hmag - G2mag - 0.08)
    AH    = 1.55 * AK
    AJ    = 2.5 * AK

    # uncertainty in AK, AJ, AH
    if Hmag_err!=None and G2mag_err!=None:
        AKerr = 0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)
        AJerr = 2.5 * AKerr
        AHerr = 1.55*0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)
    else:
        AKerr = None
        AJerr = None
        AHerr = None

    # intrinsic K-, J-, and H-band magnitudes
    H0 = Hmag - AH
    if Hmag_err!=None and G2mag_err!=None:
        H0_err = np.sqrt((1.-1.55*0.918)**2.*Hmag_err**2. + ()**2.*G2mag_err**2.)
    if Jmag!=None:
        J0 = Jmag - AJ
        if Jmag_err!=None and AJerr!=None:
            J0_err = np.sqrt(Jmag_err**2. + AJerr**2.)
    if Kmag!=None:
        K0 = Kmag - AK
        if Kmag_err!=None and AKerr!=None:
            K0_err = np.sqrt(Kmag_err**2. + AKerr**2.)

    return J0,J0_err,H0,H0_err,K0,K0_err

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
