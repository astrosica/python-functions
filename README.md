# Python functions

Possibly useful functions written in Python.

## Polarization

Computes the following using input Stokes maps:
* [x] polarized intensity
* [x] de-biased polarized intensity
* [x] polarization gradient (with and without cross-terms)
* [x] normalized polarization gradient
* [x] argument of polarization gradient
* [x] radial and tangential components of polarization gradient

## FITS files

* [x] construct the frequency axis of a 3D data cube using the header
* [x] construct a coordinate grid using the header
* [x] transform a coordinate grid from equatorial to Galactic coordinates
* [x] transform a coordinate grid from Galactic to equatorial coordinates
* [x] reproject a 2D image to a specified header
* [ ] reproject a 3D image to a specified header
* [x] reproject a 2D image from equatorial to Galactic coordinates
* [x] reproject a 3D image from equatorial to Galactic coordinates
* [ ] reproject a 2D image from Galactix to equatorial coordinates
* [ ] reproject a 3D image from Galactix to equatorial coordinates
* [x] transform a 3D header to a 2D header
* [x] slice a 3D data cube along its third axis and saves each 2D image as a separate image file

## Sky Coordinates

* [x] convert Right Ascension and Declination coordinates from decimal degrees to the sexagismal system
* [x] convert Right Ascension and Declination coordinates from the sexagismal system to decimal degrees
* [x] convert Right Ascension and Declination coordinates from the sexagismal in the B1959 system to decimal degrees in the J2000 system
* [x] match two sets of catalogue positions in equatorial coordinates

## Photometry

* [x] convert photometric magnitude uncertainty to signal-to-noise ratio
* [x] convert photometric signal-to-noise ratio to magnitude uncertainty
* [x] convert AB magnitudes to the Vega magnitude scale
* [x] compute photometric colour and its uncertainty

## Rolling Hough Transform
See the [Rolling Hough Transform](https://github.com/seclark/RHT).
* [x] execute the RHT on all FITS files within a given directory
* [x] add contents of a FITS header to an RHT file
* [x] collect all RHT angles for each pixel in the image plane
* [x] compute the distribution in RHT angle differences between two images
* [x] construct the RHT backprojection allowing one to mask specific angles

## Miscellaneous

* [x] convolve a 2D image to a specified angular resolution using FFT convolution
* [x] mask a 2D image based on a specified signal-to-noise threshold
* [x] map angles in radians from \[0,2*pi) to \[0,pi) (i.e., to a half-polar plot)
* [x] map angles in degrees from \[0,360) to \[0,180) (i.e., to a half-polar plot)
* [x] compute the spatial gradients of a 2D image
* [x] mask a 2D image and interpolates over invalid numbers
