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
    AKerr = 0.918 * np.sqrt((Hmag_err)**2. + (G2mag_err)**2.)

    # scaling relations from Majewski
    AJ    = 2.5 * AK
    AJerr = 2.5 * Akerr

    # scaling relations from Majewski
    AH    = 1.55 * AK
    AHerr = 1.55 * Akerr

    return AJ, AJerr, AH, AHerr, AK, AKerr

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
