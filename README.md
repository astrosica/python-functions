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
* [x] constructs a coordinate grid using the header
* [x] transforms a coordinate grid from equatorial to Galactic coordinates
* [x] transforms a coordinate grid from Galactic to equatorial coordinates
* [x] reprojects a 2D image to a specified header
* [ ] reprojects a 3D image to a specified header
* [x] reprojects a 2D image from equatorial to Galactic coordinates
* [x] reprojects a 3D image from equatorial to Galactic coordinates
* [ ] reprojects a 2D image from Galactix to equatorial coordinates
* [ ] reprojects a 3D image from Galactix to equatorial coordinates
* [x] transforms a 3D header to a 2D header
* [x] slices a 3D data cube along its third axis and saves each 2D image as a separate image file

## Rolling Hough Transform
See the [Rolling Hough Transform](https://github.com/seclark/RHT).
* [x] executes the RHT on all FITS files within a given directory
* [x] collects all RHT angles for each pixel in the image plane
* [x] computes the distribution in RHT angle differences between two images
* [x] constructs the RHT backprojection allowing one to mask specific angles

## Miscellaneous

* [x] convolves a 2D image to a specified angular resolution using FFT convolution
* [x] masks a 2D image based on a specified signal-to-noise threshold
* [x] maps angles in radians from \[0,2*pi) to \[0,pi) (i.e., half-polar plot)
* [x] maps angles in degrees from \[0,360) to \[0,180) (i.e., half-polar plot)
* [x] computes the spatial gradients of a 2D image
