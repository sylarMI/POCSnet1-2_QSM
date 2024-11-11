import numpy as np
import math
import scipy
from scipy import ndimage
from utils import *
from QSM_func import *

class Data_loaders_bgrm():
    def __init__(self, opt):
        self.root_dir = opt['train_path']
        self.batch_size = opt['batch_size']
        self.img1_path, self.img2_path, self.label_path = list_simu_data(opt)      
        self.data_size = len(self.label_path)
        self.img_height = opt['patch_size'][0] #define the dims of image, image will be resize with this dim
        self.img_width = opt['patch_size'][1]
        self.img_depth = opt['patch_size'][2]
        self.shuffled_idx = np.arange(self.data_size)
        self.absolute_ind = 0
        
    #shuffle the order of all samples
    def shuffle_all(self):
        np.random.shuffle(self.shuffled_idx)
        return self

    '''load data one by one'''
    def next(self, index, opt):
        
        '''initialize image and label arrays'''
        input_1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        input_2 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 1), dtype='float32')
        label = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
            
        for i in range(len(index)):
            idx = index[i]
            input1_name, input2_name, label_name = self.img1_path[idx], self.img2_path[idx], self.label_path[idx] 
            input1_tmp, input2_tmp = open_img_seg(input1_name, input2_name)
            label_tmp,_           = open_img_seg(label_name)

            if opt['rnd_crop']: # randomize certain batches in one epoch
                randint_height = np.random.randint(input1_tmp.shape[0] - self.img_height)
                randint_width  = np.random.randint(input1_tmp.shape[1] - self.img_width)            
                randint_depth  = np.random.randint(input1_tmp.shape[2] - self.img_depth)

                input1_tmp_c   = input1_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth]
                input2_tmp_c = input2_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 
                label_tmp_c   = label_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 

                input_1[i]  = input1_tmp_c
                input_2[i]  = input2_tmp_c*100
                label[i] = np.tanh(10*label_tmp_c)         
            else:
                
                input_1[i,...] = input1_tmp
                input_2[i,...,0] = input2_tmp
                label[i,...,0] = label_tmp                  

        return input_1, input_2, label

class Data_loaders_dpinv():
    def __init__(self, opt):
        self.root_dir = opt['train_path']
        self.batch_size = opt['batch_size']
        self.img1_path, self.img2_path, self.label_path = list_simu_data(opt)      
        self.data_size = len(self.label_path)
        self.img_height = opt['patch_size'][0] #define the dims of image, image will be resize with this dim
        self.img_width = opt['patch_size'][1]
        self.img_depth = opt['patch_size'][2]
        self.shuffled_idx = np.arange(self.data_size)
        self.absolute_ind = 0
        
    #shuffle the order of all samples
    def shuffle_all(self):
        np.random.shuffle(self.shuffled_idx)
        return self

    '''load data one by one'''
    def next(self, index, opt):
        
        '''initialize image and label arrays'''
        input_1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
        input_2 = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
        label = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
            
        for i in range(len(index)):
            idx = index[i]
            input1_name, input2_name, label_name = self.img1_path[idx], self.img2_path[idx], self.label_path[idx] 
            input1_tmp, input2_tmp = open_img_seg(input1_name, input2_name)
            label_tmp,_           = open_img_seg(label_name)

            if opt['rnd_crop']: # randomize certain batches in one epoch
                randint_height = np.random.randint(input1_tmp.shape[0] - self.img_height)
                randint_width  = np.random.randint(input1_tmp.shape[1] - self.img_width)            
                randint_depth  = np.random.randint(input1_tmp.shape[2] - self.img_depth)

                input1_tmp_c   = input1_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth]
                input2_tmp_c = input2_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 
                label_tmp_c   = label_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 

                input_1[i]  = input1_tmp_c
                input_2[i]  = input2_tmp_c*100
                label[i] = np.tanh(10*label_tmp_c)         
            else:
                
                input_1[i] = input1_tmp[...,np.newaxis]
                input_2[i] = input2_tmp[...,np.newaxis]
                label[i] = label_tmp[...,np.newaxis]
        
        
        if opt['is_aug']:
            for d in range(1,len(index)):
                if np.random.randint(2):
                    rnd = np.random.rand()
                    input_2[d] = rnd * input_2[d-1] + (1-rnd) * input_2[d]
                    label[d] = rnd * label[d-1] + (1-rnd) * label[d]
                    
            for d in range(len(index)):
                rnd = max(np.amax(label[d]),-np.amin(label[d]))
                rnd = max(0.01, rnd)
                s_rnd = 1/rnd * np.random.rand()

                if s_rnd * max(np.amax(label[d]),-np.amin(label[d])) > 1:
                    s_rnd = 1

                input_2[d] = s_rnd * input_2[d]
                label[d] = s_rnd * label[d]                    

        return input_1, input_2, label    
    
class Data_loaders_invivo1():
    def __init__(self, opt):
        self.root_dir = opt['train_path']
        self.batch_size = opt['batch_size']
        self.img1_path, self.img2_path, self.label_path = list_single_data(opt)
#         self.img1_path, self.img2_path, self.label_path = list_invivo_batch(opt)
#         self.img1_path, self.img2_path, self.label_path = list_invivo_data(opt)
#         self.img1_path, self.img2_path, self.label_path = list_mb_data(opt)
        self.data_size = len(self.label_path)
        self.img_height = opt['patch_size'][0] #define the dims of image, image will be resize with this dim
        self.img_width = opt['patch_size'][1]
        self.img_depth = opt['patch_size'][2]

    '''load data one by one'''
    def next(self, index, opt):
        
        '''initialize image and label arrays'''
        input1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 12), dtype='float32')
        input_1m = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        input_2 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 1), dtype='float32')
        label = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
        M = np.empty((len(index), self.img_height, self.img_width, self.img_depth, len(opt['rad'])), dtype='float32')
        msk_t = np.zeros((self.img_height, self.img_width, self.img_depth), dtype='float32')
            
        for i in range(len(index)):
            idx = index[i]
            input1_name, input2_name, label_name = self.img1_path[idx], self.img2_path[idx], self.label_path[idx] 
            input1_tmp, input2_tmp = open_img_seg(input1_name, input2_name)
            label_tmp,_           = open_img_seg(label_name)

            if opt['rnd_crop']: # randomize certain batches in one epoch
                randint_height = np.random.randint(input1_tmp.shape[0] - self.img_height)
                randint_width  = np.random.randint(input1_tmp.shape[1] - self.img_width)            
                randint_depth  = np.random.randint(input1_tmp.shape[2] - self.img_depth)

                input1_tmp_c   = input1_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth]
                input2_tmp_c = input2_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 
                label_tmp_c   = label_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 

                input_1[i]  = input1_tmp_c
                input_2[i]  = -input2_tmp_c*input1_tmp_c*10
                label[i] = label_tmp_c *input1_tmp_c*10   
            else:       
                p1 = 0#2 4
                p2 = 0#2 4
                p3 = 0#6 5
                
                input1[i,...] = np.pad(input1_tmp,((p1,p1),(p2,p2),(p3,p3),(0,0)),'constant')
                for n in range(len(opt['rad'])):
                    M[i,...,n] = input1[i,...,12-opt['rad'][n]]
                    if n is 0:
                        input_1m[i,...,n:n+1] = M[i,...,n:n+1]
                    else:
                        input_1m[i,...,n:n+1] = M[i,...,n:n+1]-M[i,...,n-1:n]
                        
                msk_t = M[...,-1]
                input_2[i,...,0] = np.pad(input2_tmp*10,((p1,p1),(p2,p2),(p3,p3)),'constant')/(2*math.pi*127*0.025)
                label[i,...,0] = np.pad(label_tmp*10,((p1,p1),(p2,p2),(p3,p3)),'constant')*msk_t
                
        return input_1m, input_2, label
    
class Data_loaders_invivo2():
    def __init__(self, opt):
        self.root_dir = opt['train_path']
        self.batch_size = opt['batch_size']
        self.img1_path, self.img2_path, self.label_path = list_single_data(opt)
#         self.img1_path, self.img2_path, self.label_path = list_invivo_batch(opt)
#         self.img1_path, self.img2_path, self.label_path = list_invivo_data(opt)
#         self.img1_path, self.img2_path, self.label_path = list_mb_data(opt)
        self.data_size = len(self.label_path)
        self.img_height = opt['patch_size'][0] #define the dims of image, image will be resize with this dim
        self.img_width = opt['patch_size'][1]
        self.img_depth = opt['patch_size'][2]
  

    '''load data one by one'''
    def next(self, index, opt):
        
        '''initialize image and label arrays'''
        input_1 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 1), dtype='float32')
        input_2 = np.empty((len(index), self.img_height, self.img_width, self.img_depth, 1), dtype='float32')
        label = np.empty((len(index), self.img_height, self.img_width, self.img_depth,1), dtype='float32')
            
        for i in range(len(index)):
            idx = index[i]
            input1_name, input2_name, label_name = self.img1_path[idx], self.img2_path[idx], self.label_path[idx] 
            input1_tmp, input2_tmp = open_img_seg(input1_name, input2_name)
            label_tmp,_           = open_img_seg(label_name)

            if opt['rnd_crop']: # randomize certain batches in one epoch
                randint_height = np.random.randint(input1_tmp.shape[0] - self.img_height)
                randint_width  = np.random.randint(input1_tmp.shape[1] - self.img_width)            
                randint_depth  = np.random.randint(input1_tmp.shape[2] - self.img_depth)

                input1_tmp_c   = input1_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth]
                input2_tmp_c = input2_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 
                label_tmp_c   = label_tmp[randint_height:self.img_height+randint_height, \
                                         randint_width:self.img_width+randint_width,randint_depth:self.img_depth+randint_depth] 

                input_1[i]  = input1_tmp_c
                input_2[i]  = -input2_tmp_c*input1_tmp_c*10
                label[i] = label_tmp_c *input1_tmp_c*10   
            else:       
                p1 = 0#2 4
                p2 = 0#2 4
                p3 = 0#6 5
                
                input_1[i,...,0] = np.pad(input1_tmp,((p1,p1),(p2,p2),(p3,p3)),'constant')
                input_2[i,...,0] = np.pad(input2_tmp,((p1,p1),(p2,p2),(p3,p3)),'constant')
                label[i,...,0] = np.pad(label_tmp,((p1,p1),(p2,p2),(p3,p3)),'constant')
                
        return input_1, input_2, label
      