# Python functions

Python functions for PhD work.

## Polarization

* [x] compute the polarized intensity and its uncertainty
* [x] compute the polarization angle and its uncertainty
* [x] compute the polarization fraction and its uncertainty
* [x] compute the de-biased polarized intensity
* [x] compute the polarization gradient (with and without cross-terms)
* [x] compute the normalized polarization gradient (with and without cross-terms)
* [x] compute the argument of polarization gradient (with and without cross-terms)
* [x] apply a mask to the argument of polarization gradient
* [x] compute the radial and tangential components of the polarization gradient
* [x] compute the angular version of the polarization gradient
* [x] compute the plane-of-sky magnetic field orientation
* [x] compute an estimate of the polarization angle dispersion function
* [x] compute the de-rotated magnetic field orientation
* [x] create a dictionary of polarization gradient arguments for each pixel
* [x] plot an image with pseudovectors overlaid
* [x] compute the rotation measure structure function (RMSF) and related parameters
* [x] compute the first moment of a one-dimensional Faraday dispersion function
* [x] compute the second moment of a one-dimensional Faraday dispersion function
* [x] compute the first moment map of a three-dimensional Faraday dispersion function
* [x] compute the second moment map of a three-dimensional Faraday dispersion function

## FITS files

* [x] construct the frequency axis of a 3D FITS file
* [x] compute the average across a data cube's third axis
* [x] compute the summation of a list of 2D FITS files
* [x] create a grid of equatorial coordinates for a FITS file
* [x] create a grid of Galactic coordinates for a FITS file
* [x] transform a coordinate grid from equatorial to Galactic coordinates
* [x] transform a coordinate grid from Galactic to equatorial coordinates
* [x] reproject one 2D FITS image to another
* [ ] reproject one 3D FITS image to another
* [x] reproject a 2D FITS image from equatorial to Galactic coordinates
* [x] reproject a 3D FITS image from equatorial to Galactic coordinates
* [ ] reproject a 2D FITS image from Galactic to equatorial coordinates
* [ ] reproject a 3D FITS image from Galactic to equatorial coordinates
* [x] reproject a HEALPix image to a standard FITS projection
* [x] creates a mask using a condition on Galactic latitude
* [x] masks a 2D FITS image in equatorial coordinates based on Galactic latitude
* [x] transform a 3D FITS header to a 2D FITS header
* [x] slice a 3D FITS data cube along its third axis and save each 2D image as a separate FITS image

## Sky Coordinates

* [x] transform Right Ascension and Declination coordinates from decimal degrees to the sexagismal system
* [x] transform Right Ascension and Declination coordinates from the sexagismal system to decimal degrees
* [x] precess Right Ascension and Declination coordinates from the sexagismal system in the B1959 equinox to decimal degrees in the J2000 equinox
* [x] match two sets of catalogue positions in equatorial coordinates
* [x] write a set of cooridinates to an annotation file for kvis
* [x] write a set of cooridinates to a region file for ds9

## Photometry

* [x] convert photometric magnitude uncertainty to signal-to-noise ratio
* [x] convert photometric signal-to-noise ratio to magnitude uncertainty
* [x] convert AB magnitudes to the Vega magnitude scale
* [x] compute photometric colour and its uncertainty
* [x] compute J-, H-, and K-band extinctions using the RJCE technique
* [x] compute J-, H-, and K-band intrinsic magnitudes using the RJCE technique
* [x] compute J-, H-, and K-band distance-corrected luminosities using Gaia parallaxes
* [x] compute distance using Gaia parallax

## Rolling Hough Transform
See the [Rolling Hough Transform](https://github.com/seclark/RHT).
* [x] execute the RHT on all FITS files within a given directory
* [x] add contents of a FITS image header to an RHT FITS file
* [x] collect all RHT angles for each spatial pixel in the image plane
* [x] compute the distribution in RHT angle differences between two images
* [x] compute the RHT backprojection allowing one to mask specific angles

## Planck

* [x] converts Planck Stokes maps from K_CMB to MJy/sr
* [x] converts Planck Stokes maps from uK to MJy/sr
* [x] remove the CMB monopole from the Planck 353 GHz Stokes maps
* [x] add the Galactic HI offset correction to the Planck 353 GHz Stokes I map

## Line Integral Convolution (LIC)

* [x] compute magnetic field orientation in IAU convention
* [x] compute LIC texture
* [x] overplot LIC texture on an image

## Colormaps

* [x] twilight colormap
* [x] Planck frequency colormap
* [x] Planck parchment colormap

## Miscellaneous

* [x] convert frequency to wavelength
* [x] convert wavelength to frequency
* [x] compute upper limit kinetic temperature from line broadening
* [x] mask a 2D FITS image in signal
* [x] mask a 2D FITS image in signal-to-noise
* [x] compute the 1D and 2D fast Fourier transform (FFT)
* [x] compute the 1D and 2D inverse fast Fourier transform (IFFT)
* [x] compute the 2D spatial gradient of an image
* [x] map angles defined on the polar plane \[0,2pi) to the half polar plane \[0,pi)
* [x] compute the angular difference between two angles defined on the half-polar plane
* [x] convolve a 2D FITS image using FFT convolution
* [x] reads in a ROHSA data file
* [x] mask and interpolate over point sources
* [x] applies a mask to a 2D FITS image and interpolates
* [x] mask an image within the boundary of a circle
* [x] mask an image within the boundary of an ellipse
* [x] mask an image betweeen two lines
* [x] masks basketweaving artefacts of the GALFACTS sensitivity map FFT

