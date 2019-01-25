import numpy as np
import json

class cvData_tfidf:
    def __init__(self, input, output, kval=10, batch_size=256):
        # stores sources or raw data
        # both, training and validation data have to avaliable
        self.__act_val = 0
        self.__tr_batch = 0
        self.__tr_val = 0
        self.rp = np.random.permutation(len(input))
        self.input = input[rp]
        self.output = output[rp]
        self.k = kval
        self.batch_size = batch_size
        self.idx_tr = list()
        self.idx_dev = list()
        self.k_size = list()
        self.in_size = len(input)
        step = self.in_size // kval
        index = 0
        for i in range(kval):
            index_end = min(index+step,self.in_size)
            self.idx_tr.append(list())
            self.idx_dev.append([index,index_end,index_end-index])
            self.k_size.append(index_end-index)
            bi = index
            while bi < index_end:
                bi_new = min(bi+self.batch_size,index_end)
                self.idx_tr[i].append([bi,bi_new,bi_new-bi])
                bi = bi_new
            index += step
        self.ended_val = list()
        
    def set_placeholders(self,pl_list):
        self.ph_x = pl_list[0]
        self.ph_y = pl_list[1]
        self.ph_w = pl_list[2]

    def init_val(self,k):
        # initialize k-th validation
        self.__act_val = k
        self.__tr_batch = 0
        self.__tr_val = 0

    
    def end_val(self,k):
        # final operations of k-th validation
        self.ended_val.append(k)

    def get_next_batch(self):
        # returns feed dictionary of one batch of training data
        ab = self.__tr_batch
        av = self.__tr_val
        self.__tr_batch += 1
        if self.__tr_batch == len(self.idx_tr[av]):
            self.__tr_batch = 0
            if av == self.__act_val:
                self.__tr_val = self.__act_val+1
            else: 
                if av == self.k:
                    self.__tr_val = 0
                else:
                    self.__tr_val += 1
                    

        d = self.idx_tr[av][ab]
        return {self.ph_x: self.input[d[0]:d[1]],
            self.ph_y: self.output[d[0]:d[1]],
            self.ph_w: d[2]}
    
    def accumulate_grad(self):
        # returns True if the previous gen_next_batch data should be used just for accumulation gradients
        # returns False if the model shoudl be learned with previously accumulated gradients 
        return False

    def get_next_dev_batch(self):
        # returns feed dictionary of one batch of dev data
        d = self.idx_dev[self.__act_val]
        return {self.ph_x: self.input[d[0]:d[1]],
            self.ph_y: self.output[d[0]:d[1]],
            self.ph_w: d[2]}

    def accumulate_dev(self):
        return False