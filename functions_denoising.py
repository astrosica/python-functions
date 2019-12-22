# Code from Luke Pratley (and adapted from http://proximity-operator.net/tutorial.html)

import matplotlib.pyplot as plt
from astropy.io import fits
import time as time
import numpy as np
import pywt

from functions_polarization import fpolgrad_crossterms, fPI

class EmptyFunction:
    def fun(self, x):      return 0
    def grad(self, x):     return 0
    def prox(self, x,tau): return x
    def dir_op(self, x):   return x
    def adj_op(self, x):   return x
    beta = 0
    

def FBPD(x_init, f=None, g=None, h=None, opt=None):

    # default inputs
    if f   is None: f = EmptyFunction()
    if g   is None: g = EmptyFunction()
    if h   is None: h = EmptyFunction()
    if opt is None: opt = {'tol': 1e-4, 'iter': 500}

    # algorithmic parameters
    tol      = opt['tol']
    max_iter = opt['iter']
    
    # step-sizes
    tau   = 2.0 / (g.beta + 2.0);
    sigma = (1.0/tau - g.beta/2.0) / h.beta;

    # initialization
    x = x_init
    y = h.dir_op(x);

    print('Running FBPD...');
    
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)

    # algorithm loop
    for it in range(0, max_iter):
    
        t = time.time()
    
        # primal forward-backward step
        x_old = x;
        x = x - tau * ( g.grad(x) + h.adj_op(y) );
        x = f.prox(x, tau);
    
        # dual forward-backward step
        y = y + sigma * h.dir_op(2*x - x_old);
        y = y - sigma * h.prox(y/sigma, 1./sigma);   

        # time and criterion
        timing[it] = time.time() - t
        criter[it] = f.fun(x) + g.fun(x) + h.fun(h.dir_op(x));
           
        # stopping rule
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
           break
        if(it % 100 == 0):
            print(str(it)+' out of '+str(max_iter)+' iterations; tol = ' + str(np.linalg.norm(x - x_old)/ np.linalg.norm(x_old)))

    criter = criter[0:it+1];
    timing = np.cumsum(timing[0:it+1]);
    
    return x, it, timing, criter

class L2_Ball:
    '''
    This class computes the proximity operator of the function

                        f(x) = gamma * i_B(x)

    When the input 'x' is an array. 'x' is processed as a single vector [DEFAULT]


     INPUTS
    ========
     x         - ND array
     epsilon   - radius of l2-ball
     data      - data that that centres the l2-ball
    '''
    
    
    def __init__(self, epsilon, data):

        if np.any(epsilon <= 0 ):
            raise Exception("'epsilon' must be positive")
        self.epsilon     = epsilon;
        self.data = data
    
    def prox(self, x, gamma):
        
        xx = np.sqrt(np.sum( np.square(np.abs(x - self.data))))
        if (xx < self.epsilon):
            p  = x
        else:
            p =  (x - self.data) * self.epsilon/xx  + self.data
        
        return p
        
    def fun(self, x):
        return 0;
    def dir_op(self, x):
        return x
    def adj_op(self, x):
        return x

class l1_norm_prox :
    
    gamma = 1
    
    def __init__(self, gamma):

        if np.any( gamma <= 0 ):
            raise Exception("'gamma' must be positive")

        self.gamma = gamma
        self.beta = 1.

    def prox(self, x, tau):

        return np.maximum(0, np.abs(x) - self.gamma * tau) * np.exp(complex(0, 1) * np.angle(x));
        
    def fun(self, x):

        return self.gamma * np.abs(x).sum();
    
    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x

class wavelets :
    
    def __init__(self, wav, levels, shape = None):

        if np.any( levels <= 0 ):
            raise Exception("'levels' must be positive")
            
        self.wav = wav
        self.levels = levels
        self.shape = shape
        self.coeff_slices = None
        self.coeff_shapes = None
        
    def forward(self, x):
        if (self.wav == "dirac"):
            return np.ndarray.flatten(x,(x.shape[0] * x.shape[1], 1))
        coeffs = pywt.wavedecn(x, wavelet=self.wav, level=self.levels, mode ='periodic')
        arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(coeffs)
        return arr
    def backward(self, x):
        if (self.wav == "dirac"):
            return np.reshape(x,self.shape)
        coeffs_from_arr = pywt.unravel_coeffs(x, self.coeff_slices, self.coeff_shapes, output_format='wavedecn')
        cam_recon = pywt.waverecn(coeffs_from_arr, wavelet=self.wav, mode ='periodic')
        return cam_recon

class dictionary:
    sizes = []
    wavelet_list = []
    
    def __init__(self, wav, levels, shape = None):

        if np.any( levels <= 0 ):
            raise Exception("'levels' must be positive")
        self.wavelet_list = []
        self.sizes = np.zeros(len(wav))
        for i in range(len(wav)):
            self.wavelet_list.append(wavelets(wav[i], levels, shape))
    
    def forward(self, x):
        out = self.wavelet_list[0].forward(x)
        self.sizes[0] = out.shape[0]
        for wav_i in range(1, len(self.wavelet_list)):
            buff = self.wavelet_list[wav_i].forward(x)
            self.sizes[wav_i] = buff.shape[0]
            out = np.concatenate((out, buff), axis=0)
        return out/ np.sqrt(len(self.wavelet_list))

    def backward(self, x):
        offset = 0
        out = 0
        for wav_i in range(len(self.wavelet_list)):
            size = self.sizes[wav_i]
            x_block = x[int(offset):int(offset + size)]
            buff = self.wavelet_list[wav_i].backward(x_block)
            out += buff / np.sqrt(len(self.wavelet_list))
            offset += size
        return out

def denoise_polgrad(Q_file,U_file,noise_sigma=1.1e-2,levels=6,stepsize=1e-3,tol=5e-5,iter=200000):
    '''
    Applies the de-noising algorithm to Stokes maps and computes the de-noised polarization gradient.

    Input
    Q_file      : directory to Stokes Q file
    U_file      : directory to Stokes U file
    noise_sigma : noise in Jy/beam on each pixel
    levels      : number of recursive wavelets
    stepsize    : step size
    tol         : the relative difference of the solution between iterations to give convergence
    iter        : the maximum numer of iterations

    Output
    Q_res       : restored Stokes Q map
    U_res       : restored Stokes U map
    polgrad_res : restored polarization gradient map
    time_t      : run time in seconds
    crit        : convergence criteria
    '''
    
    Q_data,Q_header = fits.getdata(Q_file,header=True)
    U_data,U_header = fits.getdata(U_file,header=True)

    width,height  = Q_data.shape
    Q_data,U_data = Q_data[0:height,0:width],U_data[0:height,0:width]
    
    wav         = ["db8","db6","db4"]
    
    wav_op = None
    wav_op = dictionary(wav, levels, Q_data.shape)
    f = L2_Ball(np.sqrt(height*width + 2.*np.sqrt(height*width))*noise_sigma, Q_data + complex(0,1.)*U_data)
    
    # If things are not working, make this input (step-size) smaller
    h = l1_norm_prox(np.max(np.abs(wav_op.forward(Q_data + complex(0, 1.) * U_data))*stepsize))
    h.dir_op = wav_op.forward
    h.adj_op = wav_op.backward

    # minimization
    QU_res, it, time, crit = FBPD(Q_data + complex(0,1.)*U_data, f, None, h, {'tol': tol, 'iter': iter});
    Q_res = np.real(QU_res)
    U_res = np.imag(QU_res)
    
    P_res       = fPI(Q_res,U_res)
    polgrad_res = fpolgrad_crossterms(Q_res,U_res)
    
    return Q_res,U_res,P_res,polgrad_res,time,crit