#!/Users/campbell/anaconda2/bin/python

import numpy as np
from scipy import signal
from PyAstronomy import pyasl

def degtosexa(ra_deg,dec_deg):
    '''
    Converts Right Ascension and Declination from decimal degrees to sexagismal format. Inputs integers, floats, lists, or arrays.
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

def sexatodeg(ra_sexa,dec_sexa):
    '''
    Converts Right Ascension and Declination from sexagismal format to decimal degrees. Inputs integers, floats,, lists, or arrays.
    '''
    
    if (isinstance(ra_sexa,str)==True):
        '''
        if input is a single coordinate.
        '''
        sexa = ra_sexa+" "+dec_sexa
        ra_deg,dec_deg = pyasl.coordsSexaToDeg(sexa)

    elif (isinstance(ra_sexa,np.ndarray)==True) or (isinstance(ra_sexa,list)==True):
        '''
        If input is an array of coordinates.
        '''
        ra_deg_list        = []
        dec_deg_list       = []
        for i in range(len(ra_sexa)):
            ra_sexa_i      = ra_sexa[i]
            dec_sexa_i     = dec_sexa[i]
            sexa_i = ra_sexa_i+" "+dec_sexa_i
            ra_deg_i,dec_deg_i = pyasl.coordsSexaToDeg(sexa_i)
            ra_deg_list.append(ra_deg_i)
            dec_deg_list.append(dec_deg_i)
        ra_deg = np.array(ra_deg_list)
        dec_deg = np.array(dec_deg_list)
     
    return ra_deg,dec_deg

def convolve(oldres_FWHM,newres_FWHM,data,header):
    '''
    Convolves data from oldres to newres.
    
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
    
    


