import os
import h5py
import numpy as np
import healpy as hp
import coord_v_convert
import astropy.wcs as wcs
from astropy.io import fits
from astropy import units as u
import montage_wrapper as montage
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
from reproject import reproject_from_healpix

def ffreqaxis(file):
    '''
    Constructs the frequency axis for a FITS file.

    Input
    file : location of FITS file

    Output
    freqaxis : frequency axis in units defined by the FITS header
    '''

    # extract header information
    header = fits.getheader(file)
    CRVAL3 = header["CRVAL3"]
    CRPIX3 = header["CRPIX3"]
    CDELT3 = header["CDELT3"]
    NAXIS3 = header["NAXIS3"]

    #construct pixel array
    freqpixels = np.arange(NAXIS3)+1

    # transform pixels to frequency
    freqaxis   =  CRVAL3 + (freqpixels-CRPIX3)*CDELT3

    return freqaxis

def fNpix(nside):
    '''
    Computes the number of pixels in a healpix map.

    Input
    nside : the number of pixels per side

    Output
    npix : the total number of pixels in the map
    '''

    npix = int(12*nside**2)

    return npix

def fRotateHealpix_GC(file,I_field,Q_field,U_field,incoord,outcoord,inepoch=2000.,outepoch=2000.):
    '''
    Rotates a Healpix FITS file between coordinate systems.

    Input
    file     : path to Healpix file
    I_field  : field of Stokes I data
    Q_field  : field of Stokes Q data
    U_field  : field of Stokes U data
    incoord  : 
    outcoord : 
    inepoch  : 
    outepoch : 

    Output
    I_rotated_data : 
    Q_rotated_data : 
    U_rotated_data : 
    '''

    I_data = hp.read_map(planck_dust_353_dir+dust_353_file,field=I_field)
    Q_data = hp.read_map(planck_dust_353_dir+dust_353_file,field=Q_field)
    U_data = hp.read_map(planck_dust_353_dir+dust_353_file,field=U_field)

    IQU_data = np.array([I_data.tolist(),Q_data.tolist(),U_data.tolist()])

    header = fits.getheader(file)
    NSIDE  = header["NSIDE"]

    IQU_rotated_data = rotate_map(IQU_data,inepoch,outepoch,incoord,outcoord,NSIDE)

    I_rotated_data = IQU_rotated_data[0]
    Q_rotated_data = IQU_rotated_data[1]
    U_rotated_data = IQU_rotated_data[2]

    return (I_rotated_data,Q_rotated_data,U_rotated_data)

def fcubeavg(filedir_in,filedir_out,write=False):
    '''
    Computes the average of a data cube along its third axis.

    Input
    filedir_in  : path to data cube to be averaged
    filedir_out : path to write output averaged data cube to

    Output
    data_avg   : two-dimensional average image
    header_avg : projected two-dimensional FITS header
    '''

    data_cube,header_cube = fits.getdata(filedir_in,header=True)

    data_avg   = np.nanmean(data_cube,axis=0)
    header_avg = fheader_3Dto2D(filedir_in,".",write=False)

    if write==True:
        fits.writeto(filedir_out,data_avg,header_avg,overwrite=True)

    return data_avg,header_avg

def faddfits(filedir_in,filedir_out,write=True):
    '''
    Computes the summation of a list of two-dimensional FITS files.

    Input
    filedir_in  : list of paths to FITS files to be summed
    filedir_out : path to write output summed FITS file
    '''

    for i in range(len(filedir_in)):
        file = filedir_in[i]
        data,header = fits.getdata(file,header=True)
        if i==0:
            data_sum = np.zeros(shape=data.shape)
        data_sum += data

    if write==True:
        fits.writeto(filedir_out,data_sum,header,overwrite=True)

def fcoordgrid_EQ(filedir):
    '''
    Creates a grid of equatorial coordinates for a FITS file in decimal degrees.
    
    Input
    filedir : path to FITS file
    
    Output
    radec_coords : equatorial coordinate grid in decimal degrees
    '''
    data,header    = fits.getdata(filedir,header=True)
    w              = wcs.WCS(header)

    # create pixel grid
    NAXIS1,NAXIS2  = header["NAXIS1"],header["NAXIS2"]
    xarray         = np.arange(NAXIS1)-0.5
    yarray         = np.arange(NAXIS2)-0.5
    xgrid,ygrid    = np.meshgrid(xarray,yarray)

    # create equatorial coordinate grid
    ragrid,decgrid = w.all_pix2world(xgrid,ygrid,0)
    radec_coords   = SkyCoord(ragrid,decgrid,frame="fk5",unit="deg")

    return radec_coords

def fcoordgrid_GAL(filedir):
    '''
    Creates a grid of Galactic coordinates for a FITS file in decimal degrees.
    Input
    filedir : path to FITS file

    Output
    lb_coords : Galactic coordinate grid in decimal degrees
    '''

    data,header   = fits.getdata(filedir,header=True)
    w             = wcs.WCS(header)

    # create pixel grid
    NAXIS1,NAXIS2 = header["NAXIS1"],header["NAXIS2"]
    xarray        = np.arange(NAXIS1)-0.5
    yarray        = np.arange(NAXIS2)-0.5
    xgrid,ygrid   = np.meshgrid(xarray,yarray)

    # create grid Galactic coordinate grid
    lgrid,bgrid   = w.all_pix2world(xgrid,ygrid,0)
    lb_coords     = SkyCoord(lgrid,bgrid,frame="galactic",unit="deg")

    return lb_coords

def fcoordgrid_EQtoGAL(filedir):
    '''
    Creates a grid of Galactic coordinates in decimal degrees for a FITS file with a native equatorial projection.

    Input
    filedir : path to FITS file

    Output
    lb_coords : Galactic coordinate grid in decimal degrees
    '''

    data,header    = fits.getdata(filedir,header=True)
    w              = wcs.WCS(header)

    # create grid in pixels
    NAXIS1,NAXIS2  = header["NAXIS1"],header["NAXIS2"]
    xarray         = np.arange(NAXIS1)-0.5
    yarray         = np.arange(NAXIS2)-0.5
    xgrid,ygrid    = np.meshgrid(xarray,yarray)

    # create grid in equatorial coordinates
    ragrid,decgrid = w.all_pix2world(xgrid,ygrid,0)
    radec_coords   = SkyCoord(ragrid,decgrid,frame="fk5",unit="deg")

    # transform to Galactic coordinates
    lb_coords      = radec_coords.galactic

    return lb_coords

def fcoordgrid_GALtoEQ(filedir):
    '''
    Creates a grid of equaorial coordinates in decimal degrees for a FITS file that with a native Galactic projection.

    Input
    filedir : path to FITS file

    Output
    radec_coords : equatorial coordinate grid in decimal degrees
    '''

    data,header    = fits.getdata(filedir,header=True)
    w              = wcs.WCS(header)

    # create grid in pixels
    NAXIS1,NAXIS2  = header["NAXIS1"],header["NAXIS2"]
    xarray         = np.arange(NAXIS1)-0.5
    yarray         = np.arange(NAXIS2)-0.5
    xgrid,ygrid    = np.meshgrid(xarray,yarray)

    # create grid in Galactic coordinates
    lgrid,bgrid    = w.all_pix2world(xgrid,ygrid,0)
    lb_coords      = SkyCoord(lgrid,bgrid,frame="galactic",unit="deg")

    # transform to equatorial coordinates
    radec_coords   = lb_coords.fk5

    return radec_coords

def freproject_2D_simple(image1_dir,image2_dir,reproj_dir,save=False,order="nearest-neighbor"):
    '''
    Reprojects one FITS image to another.

    Input
    image1_dir : directory to image that will be reprojected
    image2_dir : directory to template image used for reprojection
    reproj_dir : the directory to which the reprojected image will be saved if save=True
    save       : if True, saves the reprojected image to the reproj_dir directory

    Output
    reproj_data : data of reprojected image
    footprint   : a mask that defines which pixels in the reprojected image have a corresponding image in the original image
    '''

    hdu1 = fits.open(image1_dir)[0] # header to be reprojected
    hdu2 = fits.open(image2_dir)[0] # reference header

    reproj_data, footprint = reproject_interp(hdu1, hdu2.header)

    if save==True:
        fits.writeto(reproj_dir,reproj_data,hdu2.header,overwrite=True)

    return (reproj_data,footprint)


def freproject_2D(image1_dir,image2_dir,reproj_dir,clean=False,save=False,order="nearest-neighbor"):
    '''
    Reprojects one FITS image to another.

    Input
    image1_dir : directory to image that will be reprojected
    image2_dir : directory to template image used for reprojection
    reproj_dir : the directory to which the reprojected image will be saved if save=True
    clean      : if True, creates new minimal headers based off inputs
    save       : if True, saves the reprojected image to the reproj_dir directory
    order      : order of interpolation (alternative options are 'bilinear', 'biquadratic', 'bicubic')

    Output
    image1_data          : data to be reprojected
    image1_header        : header of image to be reprojected
    image1_data_reproj   : data of reprojected image
    image1_header_reproj : header of image1 used for reprojection (if clean=True header content is minimal)
    image2_data          : data of image used for reprojection
    image2_header_reproj : header of image2 used for reprojection (if clean=True header content is minimal)
    footprint            : a mask that defines which pixels in the reprojected image have a corresponding image in the original image
    '''

    image1_data,image1_header=fits.getdata(image1_dir,header=True)
    image2_data,image2_header=fits.getdata(image2_dir,header=True)

    if clean==True:
        image1_header_clean = fits.Header.fromkeys(["NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CRPIX1", "CRVAL1", "CDELT1", 
                                                    "CTYPE2", "CRPIX2", "CRVAL2", "CDELT2"])
        image2_header_clean = fits.Header.fromkeys(["NAXIS", "NAXIS1", "NAXIS2", "CTYPE1", "CRPIX1", "CRVAL1", "CDELT1", 
                                                    "CTYPE2", "CRPIX2", "CRVAL2", "CDELT2"])

        image1_header_clean["NAXIS"]  = 2
        image1_header_clean["NAXIS1"] = image1_header['NAXIS1']
        image1_header_clean["NAXIS2"] = image1_header['NAXIS2']
        image1_header_clean["CTYPE1"] = image1_header['CTYPE1']
        image1_header_clean["CRPIX1"] = image1_header['CRPIX1']
        image1_header_clean["CRVAL1"] = image1_header['CRVAL1']
        image1_header_clean["CDELT1"] = image1_header['CDELT1']
        image1_header_clean["CTYPE2"] = image1_header['CTYPE2']
        image1_header_clean["CRPIX2"] = image1_header['CRPIX2']
        image1_header_clean["CRVAL2"] = image1_header['CRVAL2']
        image1_header_clean["CDELT2"] = image1_header['CDELT2']

        image2_header_clean["NAXIS"]  = 2
        image2_header_clean["NAXIS1"] = image2_header['NAXIS1']
        image2_header_clean["NAXIS2"] = image2_header['NAXIS2']
        image2_header_clean["CTYPE1"] = image2_header['CTYPE1']
        image2_header_clean["CRPIX1"] = image2_header['CRPIX1']
        image2_header_clean["CRVAL1"] = image2_header['CRVAL1']
        image2_header_clean["CDELT1"] = image2_header['CDELT1']
        image2_header_clean["CTYPE2"] = image2_header['CTYPE2']
        image2_header_clean["CRPIX2"] = image2_header['CRPIX2']
        image2_header_clean["CRVAL2"] = image2_header['CRVAL2']
        image2_header_clean["CDELT2"] = image2_header['CDELT2']

        image1_header_reproj          = image1_header_clean
        image2_header_reproj          = image2_header_clean

    else:
        image1_header_reproj          = image2_header
        image2_header_reproj          = image2_header

    # perform reprojection
    image1_data_reproj,footprint = reproject_interp((image1_data, image1_header_reproj), image2_header_reproj,order=order)

    if save==True:
        fits.writeto(reproj_dir,image1_data_reproj,image1_header_reproj,overwrite=True)

    return (image1_data,image1_header,image1_data_reproj,image1_header_reproj,image2_data,image2_header_reproj,footprint)

def freproj2D_EQ_GAL(filedir_in,filedir_out):

    '''
    Reprojects a two-dimensional FITS image from equatorial to Galactic coordinates using Montage.

    Input
    filedir_in   : input file in equatorial coordinates
    filedir_out  : output file in Galactic coordinates

    Output
    saves the reprojected FITS image to the input path filedir_out
    '''

    # extract data and headers
    data_EQ,header_EQ                   = fits.getdata(filedir_in,header=True)
    w_EQ                                = wcs.WCS(fits.open(filedir_in)[0].header)
    header_EQ_NAXIS1,header_EQ_NAXIS2   = header_EQ["NAXIS1"],header_EQ["NAXIS2"]

    # change WCS from equatorial to Galactic
    header_GAL_CTYPE1,header_GAL_CTYPE2 = ("GLON-TAN","GLAT-TAN")
    header_GAL_CUNIT1,header_GAL_CUNIT2 = ("deg","deg")
    header_GAL_CROTA1,header_GAL_CROTA2 = (0,0)

    ############################## make Galactic footprint larger ##############################
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (6000,6000)                                       # N1
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (3000,6500)                                       # N2
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (4000,7500)                                       # N3
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (6000,5500)                                       # N4
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (6000,6500)                                       # S1
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (8000,4000)                                       # S2
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (6000,7000)                                       # S3
    #header_GAL_NAXIS1,header_GAL_NAXIS2 = (8000,4000)                                       # S4
    ############################################################################################

    header_GAL_CRPIX1,header_GAL_CRPIX2 = header_GAL_NAXIS1/2.,header_GAL_NAXIS2/2.
    crpix1_GAL,crpix2_GAL               = (header_GAL_NAXIS1*0.5,header_GAL_NAXIS2*0.5)

    crpix1_EQ,crpix2_EQ                 = header_EQ_NAXIS1/2.,header_EQ_NAXIS2/2.
    crpix1_crpix2_radec                 = w_EQ.all_pix2world(crpix1_EQ,crpix2_EQ,0)
    crpix1_ra,crpix2_dec                = np.float(crpix1_crpix2_radec[0]),np.float(crpix1_crpix2_radec[1])

    # transform center pixel values from (ra,dec) to (l,b)
    coords_EQ                           = SkyCoord(ra=crpix1_ra*u.degree, dec=crpix2_dec*u.degree, frame="fk5")
    header_GAL_CRVAL1,header_GAL_CRVAL2 = (coords_EQ.galactic.l.deg,coords_EQ.galactic.b.deg)

    header_GAL_CDELT1     = header_EQ["CDELT1"]
    header_GAL_CDELT2     = header_EQ["CDELT2"]

    # write GAL header
    data_GAL              = np.zeros(shape=(header_GAL_NAXIS2,header_GAL_NAXIS1))
    header_GAL            = fits.PrimaryHDU(data=data_GAL).header
    header_GAL["NAXIS"]   = 2
    header_GAL["NAXIS1"]  = header_GAL_NAXIS1
    header_GAL["NAXIS2"]  = header_GAL_NAXIS2
    # NAXIS1
    header_GAL["CTYPE1"]  = header_GAL_CTYPE1
    header_GAL["CRPIX1"]  = header_GAL_CRPIX1
    header_GAL["CRVAL1"]  = header_GAL_CRVAL1
    header_GAL["CDELT1"]  = header_GAL_CDELT1
    header_GAL["CROTA1"]  = header_GAL_CROTA1
    # NAXIS2
    header_GAL["CTYPE2"]  = header_GAL_CTYPE2
    header_GAL["CRPIX2"]  = header_GAL_CRPIX2
    header_GAL["CRVAL2"]  = header_GAL_CRVAL2
    header_GAL["CDELT2"]  = header_GAL_CDELT2
    header_GAL["CROTA2"]  = header_GAL_CROTA2
    # other
    header_GAL["EQUINOX"] = 2000.
    header_GAL["CUNIT1"]  = header_GAL_CUNIT1
    header_GAL["CUNIT2"]  = header_GAL_CUNIT2

    # perform reprojection with Montage
    header_file  = "/Users/campbell/Documents/PhD/data/GALFACTS/N1/GAL/header_GAL.fits"
    mheader_file = "/Users/campbell/Documents/PhD/data/GALFACTS/N1/GAL/mheader_GAL.txt"
    fits.writeto(header_file,data_GAL,header_GAL,overwrite=True)
    montage.mGetHdr(header_file,mheader_file)
    os.remove(header_file)
    montage.reproject(filedir_in,filedir_out,header=mheader_file)

def freproj3D_EQ_GAL(filedir_in,filedir_out,header_file):

    '''
    Reprojects a three-dimensional FITS image from equatorial to Galactic coordinates.

    Input
    filedir_in   : input file in equatorial coordinates
    filedir_out  : output file in Galactic coordinates
    header_file  : contains the reference header used for reprojection

    Output
    saves the reprojected FITS image to the input path filedir_out
    '''

    # extract data and headers
    data_EQ_3D,header_EQ_3D                                     = fits.getdata(filedir_in,header=True)
    header_EQ_3D_NAXIS1,header_EQ_3D_NAXIS2,header_EQ_3D_NAXIS3 = header_EQ_3D["NAXIS1"],header_EQ_3D["NAXIS2"],header_EQ_3D["NAXIS3"]
    header_EQ_3D_CTYPE1,header_EQ_3D_CTYPE2,header_EQ_3D_CTYPE3 = header_EQ_3D["CTYPE1"],header_EQ_3D["CTYPE2"],header_EQ_3D["CTYPE3"]
    header_EQ_3D_CRPIX1,header_EQ_3D_CRPIX2,header_EQ_3D_CRPIX3 = header_EQ_3D["CRPIX1"],header_EQ_3D["CRPIX2"],header_EQ_3D["CRPIX3"]
    header_EQ_3D_CRVAL1,header_EQ_3D_CRVAL2,header_EQ_3D_CRVAL3 = header_EQ_3D["CRVAL1"],header_EQ_3D["CRVAL2"],header_EQ_3D["CRVAL3"]
    header_EQ_3D_CDELT1,header_EQ_3D_CDELT2,header_EQ_3D_CDELT3 = header_EQ_3D["CDELT1"],header_EQ_3D["CDELT2"],header_EQ_3D["CDELT3"]

    # change WCS from equatorial to Galactic
    header_GAL_3D_CTYPE1,header_GAL_3D_CTYPE2 = ("GLON-TAN","GLAT-TAN")
    header_GAL_3D_CUNIT1,header_GAL_3D_CUNIT2 = ("deg","deg")
    header_GAL_3D_CROTA1,header_GAL_3D_CROTA2 = (0,0)

    ############################## make Galactic footprint larger ###################################
    #header_GAL_3D_NAXIS1,header_GAL_3D_NAXIS2 = (6000,6000)                                      # N1
    #header_GAL_2D_NAXIS1,header_GAL_2D_NAXIS2 = (3000,6500)                                      # N2
    #header_GAL_2D_NAXIS1,header_GAL_2D_NAXIS2 = (4000,7500)                                      # N3
    #################################################################################################

    header_GAL_3D_CRPIX1,header_GAL_3D_CRPIX2 = header_GAL_3D_NAXIS1*0.5,header_GAL_3D_NAXIS2*0.5
    crpix1_GAL_3D,crpix2_GAL_3D               = (int(header_GAL_3D_NAXIS1*0.5),int(header_GAL_3D_NAXIS2*0.5))

    w_EQ_3D                                   = wcs.WCS(fits.open(filedir_in)[0].header)

    crpix1_EQ_2D,crpix2_EQ_2D  = header_EQ_3D_NAXIS1*0.5,header_EQ_3D_NAXIS2*0.5
    crpix1_crpix2_radec_2D     = w_EQ_3D.all_pix2world(crpix1_EQ_2D,crpix2_EQ_2D,0,0)
    crpix1_ra_2D,crpix2_dec_2D = np.float(crpix1_crpix2_radec_2D[0]),np.float(crpix1_crpix2_radec_2D[1])

    # transform center pixel values from (ra,dec) to (l,b)
    coords                                    = SkyCoord(ra=crpix1_ra_2D*u.degree, dec=crpix2_dec_2D*u.degree, frame='fk5')
    header_GAL_3D_CRVAL1,header_GAL_3D_CRVAL2 = (coords.galactic.l.deg,coords.galactic.b.deg)

    # change CDELTs from equatorial to Galactic values
    header_GAL_3D_CDELT1,header_GAL_3D_CDELT2 = (header_EQ_3D_CDELT1,header_EQ_3D_CDELT2)

    # create 3D GAL header
    data_GAL_3D                                                             = np.zeros(shape=data_EQ_3D.shape)
    header_GAL_3D                                                           = fits.PrimaryHDU(data=data_GAL_3D).header
    header_GAL_3D["CTYPE1"],header_GAL_3D["CTYPE2"],header_GAL_3D["CTYPE3"] = header_GAL_3D_CTYPE1,header_GAL_3D_CTYPE2,header_EQ_3D_CTYPE3
    header_GAL_3D["NAXIS1"],header_GAL_3D["NAXIS2"],header_GAL_3D["NAXIS3"] = header_GAL_3D_NAXIS1,header_GAL_3D_NAXIS2,header_EQ_3D_NAXIS3
    header_GAL_3D["CRPIX1"],header_GAL_3D["CRPIX2"],header_GAL_3D["CRPIX3"] = header_GAL_3D_CRPIX1,header_GAL_3D_CRPIX2,header_EQ_3D_CRPIX3
    header_GAL_3D["CRVAL1"],header_GAL_3D["CRVAL2"],header_GAL_3D["CRVAL3"] = header_GAL_3D_CRVAL1,header_GAL_3D_CRVAL2,header_EQ_3D_CRVAL3
    header_GAL_3D["CDELT1"],header_GAL_3D["CDELT2"],header_GAL_3D["CDELT3"] = header_GAL_3D_CDELT1,header_GAL_3D_CDELT2,header_EQ_3D_CDELT3
    header_GAL_3D["CUNIT1"],header_GAL_3D["CUNIT2"]                         = header_GAL_3D_CUNIT1,header_GAL_3D_CUNIT2
    header_GAL_3D["CROTA1"],header_GAL_3D["CROTA2"],header_GAL_3D["CROTA3"] = 0.0,0.0,0.0

    header_file  = "/Users/campbell/Documents/PhD/data/GALFA-HI/N1/v_-10_+10_kms/GAL/header_GAL.fits"
    mheader_file = "/Users/campbell/Documents/PhD/data/GALFA-HI/N1/v_-10_+10_kms/GAL/mheader_GAL.txt"
    fits.writeto(header_file,data_GAL_3D,header_GAL_3D,overwrite=True)
    montage.mGetHdr(header_file,mheader_file)
    os.remove(header_file)
    montage.reproject_cube(filedir_in,filedir_out,header=mheader_file,clobber=True)

def freproj_fromHEALPix(healpix_file,fits_file,output_file,coord="G",nested=False,write=True):
    '''
    Reprojects a HEALPix image to a standard FITS projection.

    Input:
    healpix_file : directory to HEALPix file to be reprojected
    fits_file    : directory to FITS file which the HEALPix image will be reprojected to
    output_file  : directory to reprojected FITS file
    coord        : coordinate system of input HEALPix image ("G" for Galactic or "C" for celestial)
    nested       : order of HEALPix data (True for nested or False for ring)
    write        : if True, writes reprojected FITS file

    Output: 
    healpix_data_reproj : reprojected HEALPix data
    footprint           : reprojection footprint
    '''

    healpix_data          = hp.read_map(healpix_file)
    fits_data,fits_header = fits.getdata(fits_file,header=True)

    healpix_data_reproj,footprint = reproject_from_healpix((healpix_data,coord),fits_header,nested=nested)

    if write==True:
        fits.writeto(output_file,healpix_data_reproj,fits_header,overwrite=True)

    return healpix_data_reproj,footprint

def freproject_HI4PI(HI4PI_input_file,FITS_file,HI4PI_output_file,VERBOSE=True):
    '''
    Reprojects the HI4PI hdf5 file to a FITS file.

    Input
    HI4PI_input_file  : directory to HI4PI file to be reprojected
    FITS_file         : directory to FITS file which the HI4PI file will be reprojected to
    HI4PI_output_file : directory to save reprojected HI4PI file
    '''

    # FITS header for reprjection
    FITS_data,FITS_header = fits.getdata(FITS_file,header=True)
    FITS_wcs    = wcs.WCS(FITS_header)

    # read in HI4PI data for image size
    f          = h5py.File(HI4PI_input_file,"r")
    HI4PI_data = f["survey"]

    # read in HI4PI hdf5 data and store each velocity channel in a dictionary
    HI4PI_data_dict = {}
    with h5py.File(HI4PI_input_file,"r") as f:
        for i in np.arange(HI4PI_data.shape[1]):
            if VERBOSE:
                print "Reading in slice {}".format(i) 
            vslice  = f["survey"][:,i]
            HI4PI_data_dict[i] = vslice

    #initialize reprojected data cube
    HI4PI_cube_reproj_data = np.ones(shape=(HI4PI_data.shape[1],4320,8640))
    # iterate through each velocity channel
    for i in np.arange(HI4PI_data.shape[1]):
        if VERBOSE:
            print "Reprojecting slice {}".format(i)
        # reproject
        healpix_data_reproj,footprint = reproject_from_healpix((HI4PI_data_dict[i],"G"),FITS_wcs,shape_out=FITS_data.shape,hdu_in=1,nested=False)
        # update cube index
        HI4PI_cube_reproj_data[i]*=healpix_data_reproj

    if VERBOSE:
        print "Saving FITS file: {}".format(HI4PI_output_file)
    # save to FITS file
    fits.writeto(HI4PI_output_file,HI4PI_cube_reproj_data,overwrite=True)

def fhighlatmask(lb_coords,blim):
    '''
    Creates a two-dimensional FITS image mask for low Galactic latitudes.

    Input
    lb_coords   : Galactic coordinate grid in decimal degrees
    blim        : lower-limit on Galactic latitude where |b|<blim will be masked

    Output
    mask : mask for low Galactic latitudes
    '''

    # construct coordinate grids
    lgrid,bgrid    = lb_coords.l.deg,lb_coords.b.deg

    # create mask
    mask           = np.ones(shape=bgrid.shape)
    ii             = np.abs(bgrid)<blim
    mask[ii]       = float("NaN")

    return mask

def fmask2DEQhighlat(filedir_in,filedir_out,blim,write=False):
    '''
    Masks a 2D FITS image in equatorial coordinates using a cut on Galactic latitude.

    Input
    filedir_in  : path to 2D FITS file in equatorial coordinates
    filedir_out : path to write masked FITS file
    blim        : minimum Galatic latitude
    write       : if True, writes the masked data to a new file (default=False)

    Output
    data_masked : masked data

    '''

    data,header = fits.getdata(filedir_in,header=True)

    # create Galactic coordinate grid
    lb_coords   = fcoordgrid_EQtoGAL(filedir_in)
    # create mask
    mask        = fhighlatmask(lb_coords,blim)
    # mask data
    data_masked = data*mask

    if write==True:
        fits.writeto(filedir_out,data_masked,header,overwrite=True)

    return data_masked

def fheader_3Dto2D(filedir_in,filedir_out,write=False):
    '''
    Transforms a three-dimensional FITS header to a two-dimensional FITS header.

    Input
    filedir_in  : path to three-dimensional FITS file
    filedir_out : path to save the two-dimensional FITS file
    write       : if True, will save the two-dimensional FITS file to the input filedir_out path (default=False)

    Output
    header : two-dimensional FITS header
    '''

    data,header = fits.getdata(filedir_in,header=True)

    header_keys = header.keys()
    header["NAXIS"]=2

    keys_3D = ["NAXIS3","CDELT3","CROTA3","CRPIX3","CRVAL3","CTYPE3"]

    for key in keys_3D:
        if key in header_keys:
            del header[key]

    if write==True:
        fits.writeto(filedir_out,data,header,overwrite=True)

    return header

def fslice3DFITS(filedir_in,dir_out,units="kms",verbose=True):
    '''
    Slices a three-dimensional FITS data cube along the third axis and saves each two-dimensional image as a separate FITS file.

    Input
    filedir_in : file directory of input FITS data cube
    dir_out    : directory where 2D image slices will be stored
    units      : units of third axis in FITS data cube
    verbose    : if True, will print each file in progress
    '''

    # extract FITS data
    data,header=fits.getdata(filedir_in,header=True)

    # create velocity axis
    third_axis       = ffreqaxis(filedir_in)

    # remove 3D information from FITS header
    header_2D = fheader_3Dto2D(filedir_in,None)

    # iterate through each channel
    for i in range(data.shape[0]):
        third_axis_i = third_axis[i]*1E-3
        data_i       = data[i]
        fname        = os.path.basename(filedir_in)+"_"+str(third_axis_i)+"_"+units+".fits"
        fdir         = dir_out+fname
        if verbose==True:
            print "writing "+fdir+"..."
        fits.writeto(fdir,data_i,header_2D,overwrite=True)