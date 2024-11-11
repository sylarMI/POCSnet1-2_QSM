import tensorflow as tf

# from keras.utils.np_utils import to_categorical
# from keras.models import Model
# from keras.layers import Input, Dense, Dropout, Flatten, Concatenate, Layer
# from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
# from keras.layers.pooling import MaxPooling3D
# from keras.layers.merge import concatenate
# from keras.layers import BatchNormalization, Activation, Add, Lambda
# from keras.layers.advanced_activations import LeakyReLU
# from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import math
from model_structures import *

def dipole_kernel(opt):
    nx,ny,nz = opt['patch_size'][0],opt['patch_size'][1],opt['patch_size'][2]
    FOVx,FOVy,FOVz=np.multiply(opt['reso'],opt['patch_size'])
    
    kx=np.arange(-np.ceil((nx-1)/2.0),np.floor((nx-1)/2.0)+1)*1.0/(FOVx/2.0)
    ky=np.arange(-np.ceil((ny-1)/2.0),np.floor((ny-1)/2.0)+1)*1.0/(FOVy/2.0)
    kz=np.arange(-np.ceil((nz-1)/2.0),np.floor((nz-1)/2.0)+1)*1.0/(FOVz/2.0)
    
    KX,KY,KZ=np.meshgrid(kx,ky,kz)
    KX=KX.transpose(1,0,2)
    KY=KY.transpose(1,0,2)
    KZ=KZ.transpose(1,0,2)
    
    K2=KX**2+KY**2+KZ**2
    
    dipole_f=1.0/3.0-KZ**2/(K2+np.finfo(float).eps)

    return dipole_f

def smv_iter_array(opt):
    ker_arr = []
    trun_arr = []
    for num in range(len(opt['rad'])):   
        nx,ny,nz=round(opt['rad'][num]/opt['reso'][0]),round(opt['rad'][num]/opt['reso'][1]),round(opt['rad'][num]/opt['reso'][2])
        nx,ny,nz = max(nx,2),max(ny,2),max(nz,2)
        ky,kx,kz = np.mgrid[-nx:nx+1,-ny:ny+1,-nz:nz+1]
        k = (kx**2/nx**2+ky**2/ny**2+kz**2/nz**2<=1)
        ker = k/np.sum(k)

        del_ker = -ker
        del_ker[nx:nx+1,ny:ny+1,nz:nz+1] = 1-ker[nx:nx+1,ny:ny+1,nz:nz+1]

        del_ker = tf.cast(del_ker,dtype = 'float32')
        y=opt['patch_size']
        del_ker = tf.pad(del_ker,[[y[0]//2-nx,y[0]//2-nx-1],[y[1]//2-ny,y[1]//2-ny-1],[y[2]//2-nz,y[2]//2-nz-1]])
        del_ker = tf.dtypes.complex(del_ker, tf.zeros_like(del_ker))
        del_ker = tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(del_ker)))
        
        del_ker_tru = tf.where(tf.abs(del_ker) <opt['thr']*tf.math.reduce_max(tf.abs(del_ker)),tf.ones_like(del_ker),tf.zeros_like(del_ker))
   
        ker_arr.append(del_ker)
        trun_arr.append(del_ker_tru)
    return ker_arr,trun_arr


def phase2qsm(x,D,opt): 
    d = D
    d = tf.cast(d, dtype = 'float32')
    d = d[tf.newaxis,tf.newaxis,:,:,:]
    d = tf.dtypes.complex(d, tf.zeros_like(d))
    
    x1 = tf.dtypes.complex(x, tf.zeros_like(x))
    x1 = tf.transpose(x1, perm=[0, 4, 1, 2, 3])
    x1k = tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(x1)))
    
    f = tf.multiply(x1k,d)
    f = tf.divide(f,tf.multiply(d,d)+opt['lbd2'])
    
    f1 = tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(f)))
    f1 = tf.transpose(f1, perm=[0, 2, 3, 4, 1])
    f1 = tf.math.real(f1)
    
    return f1

def phaseinv(x,D,opt): 
    d = D
    d = tf.cast(d, dtype = 'float32')
    d = d[tf.newaxis,tf.newaxis,:,:,:]
    d = tf.dtypes.complex(d, tf.zeros_like(d))
    
    x1 = tf.dtypes.complex(x, tf.zeros_like(x))
    x1 = tf.transpose(x1, perm=[0, 4, 1, 2, 3])
    x1k = tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(x1)))
    
    #f = tf.multiply(x1k,d)
    f = tf.divide(x1k,tf.multiply(d,d)+opt['lbd2'])
    
    f1 = tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(f)))
    f1 = tf.transpose(f1, perm=[0, 2, 3, 4, 1])
    f1 = tf.math.real(f1)
    
    return f1

# def qsm2phase(y,opt):
#     d = dipole_kernel(opt)
#     d = tf.cast(d, dtype = 'float32')
#     d = d[tf.newaxis,tf.newaxis,:,:,:]
#     d = tf.dtypes.complex(d, tf.zeros_like(d))
    
#     x1 = tf.dtypes.complex(y, tf.zeros_like(y))
#     x1 = tf.transpose(x1, perm=[0, 4, 1, 2, 3])
#     x1k = tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(x1)))

#     fk1 = tf.multiply(x1k, d)

#     f1 = tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(fk1)))
#     f1 = tf.transpose(f1, perm=[0, 2, 3, 4, 1])
#     f1 = tf.math.real(f1)
    
#     return f1

def cg(M,x,smv,opt):
    x1 = tf.dtypes.complex(x, tf.zeros_like(x))
    x1 = tf.transpose(x1, perm=[0, 4, 1, 2, 3])
    x1k = tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(x1)))
    phs = tf.zeros_like(x, dtype = 'complex64')
    phs = tf.transpose(phs, perm=[0, 4, 1, 2, 3])
    for n in range(len(opt['rad'])):

        ker = smv[n] 
        ker = ker[tf.newaxis,tf.newaxis,:,:,:]
        f1 = tf.multiply(x1k,ker)
        f1 = tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(f1)))

        Mm = tf.dtypes.complex(M[...,n:n+1],tf.zeros_like(M[...,n:n+1]))
        Mm = tf.transpose(Mm,perm=[0, 4, 1, 2, 3])
            
        f1 = tf.multiply(Mm,f1)
        
        f1 = tf.signal.fftshift(tf.signal.fft3d(tf.signal.fftshift(f1)))
        f1 = tf.multiply(f1,tf.math.conj(ker))
        
        phs = phs + f1
    f1 = tf.signal.ifftshift(tf.signal.ifft3d(tf.signal.ifftshift(phs)))
    f1 = tf.transpose(f1,perm=[0, 2, 3, 4, 1])
    f1 = tf.math.real(f1)   
    Ax = f1
    return Ax


def conjgrad_br_trun(M,L,T,Lref,smv,trun,opt): 
    Ax = cg(M,L,smv,opt) + opt['lbd1']*L+opt['lbd0']*cg(M,L,trun,opt)
    b  = cg(M,T,smv,opt) + opt['lbd1']*Lref
    r = b - Ax
    p = r
    rsold = tf.math.reduce_sum(r*r)
    
    for i in range(opt['c_iter']):
        Ap = cg(M, p, smv, opt) + opt['lbd1']*p+opt['lbd0']*cg(M,p,trun,opt)
        alpha = rsold / tf.math.reduce_sum(p*Ap)
        L = L + alpha*p
        r = r - alpha*Ap
        rsnew = tf.math.reduce_sum(r*r)
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return L

def conjgrad_br_trun_grad(M,L,T,Lref,smv,trun,opt):
    Ax = cg(M,L,smv,opt) + opt['lbd1']*L+opt['lbd0']*cg(M,L,trun,opt)
    b  = opt['lbd1']*Lref
    r = b - Ax
    p = r
    rsold = tf.math.reduce_sum(r*r)
    
    for i in range(opt['c_iter']):
        Ap = cg(M, p, smv, opt) + opt['lbd1']*p+opt['lbd0']*cg(M,p,trun,opt)
        alpha = rsold / tf.math.reduce_sum(p*Ap)
        L = L + alpha*p
        r = r - alpha*Ap
        rsnew = tf.math.reduce_sum(r*r)
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return L
