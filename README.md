# Python functions

Useful functions written in Python. Strikethroughs indicate those that are not yet available.

## Polarization

Compute the following using input Stokes maps:
* polarized intensity
* polarization gradient
* argument of polarization gradient
* radial and tangential components of polarization gradient

## FITS files

* construct the frequency axis using the FITS header
* constructs a coordinate grid using the FITS header
* transforms a coordinate grid from equatorial to Galactic coordinates
* transforms a coordinate grid from Galactic to equatorial coordinates
* reprojects a 2D ~~or 3D~~ image to a specified header
* reprojects a 2D or 3D image from equatorial to Galactic coordinates
* ~~reprojects a 2D or 3D image from Galactix to equatorial coordinates~~
* transforms a 3D FITS header to 2D
* slices a 3D FITS cube along the third axis and saves each 2D image as a separate FITS file

## Rolling Hough Transform

* executes the RHT on all FITS files within a given directory
* creates a dictionary of RHT angles for each pixel in the image plane
* computes the distribution in RHT angle differences between two images
* constructs the RHT backprojection allowing one to mask specific angles

## Miscellaneous

* convolves a 2D FITS image to a specified angular resolution using FFT convolution
* masks a 2D image based on signal-to-noise
* maps angles in radians from \[0,2*pi) to \[0,pi) (i.e., half-polar plot)
* maps angles in degrees from \[0,360) to \[0,180) (i.e., half-polar plot)
