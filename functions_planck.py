def fPlanc_KCMB_MJysr(data,freq):
	'''
	Converts Planck Stokes maps from K_CMB to MJy/sr.
	See Table 6 from Planck IX (2013).
	'''

	if freq==100:
		fac = 244.1   # MJy/sr / K_CMB
	elif freq==143:
		fac = 371.74  # MJy/sr / K_CMB
	elif freq==217:
		fac = 483.690 # MJy/sr / K_CMB
	elif freq==353:
		fac = 287.450 # MJy/sr / K_CMB
	elif freq==545:
		fac = 58.04   # MJy/sr / K_CMB
	elif freq==857:
		fac = 2.27    # MJy/sr / K_CMB

	data_MJysr = data*fac

	return data_MJysr

def fPlanc_uK_MJysr(data,freq):
	'''
	Converts Planck Stokes maps from uK to MJy/sr.
	See Table 5 from Planck IX (2013).
	'''

	if freq==100:
		fac = 0.0032548074*1E6   # uK/K_CMB
	elif freq==143:
		fac = 0.0015916707*1E6   # uK/K_CMB
	elif freq==217:
		fac = 0.00069120334*1E6  # uK/K_CMB
	elif freq==353:
		fac = 0.00026120163*1E6  # uK/K_CMB
	elif freq==545:
		fac = 0.00010958025*1E6  # uK/K_CMB
	elif freq==857:
		fac = 0.000044316316*1E6 # uK/K_CMB

	data_MJysr = data/fac

	return data_MJysr

def fPlanckCMBmonopole(data,unit="K"):
	'''
	Removes the CMB monopole from the Planck 353 GHz Stokes maps.
	Offset corrections from Planck III (2018).

	Input
	data : Planck Stokes map
	unit : dimensions of Stokes map [K or MJysr]
	'''

	if unit=="K":
		offset = 452E-6
	elif unit=="MJysr":
		offset = 0.13

	data_corr = data - offset

	return data_corr

def fPlanckHIcorr(data,unit="K"):
	'''
	Adds the Galactic HI offset correction to the Planck 353 GHz Stokes I map.
	Offset correction from Planck XII (2018).

	Input
	data : Planck Stokes I map
	unit : dimensions of Stokes map [K or MJysr]
	'''

	if unit=="K":
		offset = 36E-6
	elif unit=="MJysr":
		fac = 287.5 # converts K_CMB to MJy/sr
		offset = 36E-6 * fac

	data_corr = data + offset

	return data_corr