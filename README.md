# Python functions

Possibly useful functions written in Python.

## Polarization

* [x] compute the polarized intensity
* [x] compute the de-biased polarized intensity
* [x] compute the polarization gradient (with and without cross-terms)
* [x] compute the normalized polarization gradient
* [x] compute the argument of polarization gradient (with and without cross-terms)
* [x] compute the radial and tangential components of polarization gradient
* [x] plot an image with pseudovectors overlaid

## FITS files

* [x] construct the frequency axis of a 3D FITS file
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
* [x] transform a 3D FITS header to a 2D FITS header
* [x] slice a 3D FITS data cube along its third axis and saves each 2D image as a separate FITS image

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

## Rolling Hough Transform
See the [Rolling Hough Transform](https://github.com/seclark/RHT).
* [x] execute the RHT on all FITS files within a given directory
* [x] add contents of a FITS image header to an RHT FITS file
* [x] collect all RHT angles for each spatial pixel in the image plane
* [x] compute the distribution in RHT angle differences between two images
* [x] compute the RHT backprojection allowing one to mask specific angles

## Miscellaneous

* [x] convolve a 2D FITS image using FFT convolution
* [x] compute the 1D and 2D fast Fourier transform (FFT)
* [x] compute the 1D and 2D inverse fast Fourier transform (IFFT)
* [x] compute the 2D spatial gradient of an image
* [x] mask a 2D FITS image in signal-to-noise
* [x] mask a 2D FITS image in signal
* [x] mask a 2D image and interpolates over invalid numbers
* [x] map angles from \[0,2*pi) to \[0,pi) (i.e., to a half-polar plot in radians)
* [x] map angles from \[0,360) to \[0,180) (i.e., to a half-polar plot in degrees)
