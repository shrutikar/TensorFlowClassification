import tensorflow as tf
import numpy as np

class EmbData:       
    def __init__(self, train_input, train_target, test_input, test_target,
        padding = 1,
        max_words = 5000):
        num_class = len(train_target[0])
        for i in range(len(train_input)) :
            for j in range(len(train_input[i])) :
                train_input[i][j] = min(train_input[i][j],max_words)
        
        for i in range(len(test_input)) :
            for j in range(len(test_input[i])) :
                test_input[i][j] = min(test_input[i][j],max_words)
                
        self.max_words = max_words
        self.train_size = len(train_input)
        self.test_size = len(test_input)
        self.batch_index = 0
        self.test_batch_index = 0
        self.accumulate = True
        self.accumulate_test = True
        self.train_sizes = dict()
        for i,sample in enumerate(train_input):
            if not len(sample) in self.train_sizes :
                self.train_sizes[len(sample)] = [i]
            else:
                self.train_sizes[len(sample)].append(i)
        self.batch_size = len(self.train_sizes)
        self.train_input = list()
        self.train_target = list()
        for size in self.train_sizes :
            inp = np.zeros([len(self.train_sizes[size]),size+2*padding])
            target = np.zeros([len(self.train_sizes[size]),len(train_target[0])])
            for i,index in enumerate(self.train_sizes[size]) :
                inp[i,padding:-padding] = train_input[index]
                target[i,:] = train_target[index]
            self.train_input.append(inp)
            self.train_target.append(target)
        self.train_weights = np.sum(self.train_target,axis=1)
        self.train_weights[self.train_weights>0] = 1/self.train_weights[self.train_weights>0]/num_class
        self.test_sizes = dict()
        for i,sample in enumerate(test_input):
            if not len(sample) in self.test_sizes :
                self.test_sizes[len(sample)] = [i]
            else:
                self.test_sizes[len(sample)].append(i)
        self.test_input = list()
        self.test_target = list()
        self.test_batch_size = len(self.test_sizes)
        for size_index,size in enumerate(self.test_sizes) :
            inp = np.zeros([len(self.test_sizes[size]),size+2*padding])
            target = np.zeros([len(self.test_sizes[size]),len(test_target[0])])            
            for i,index in enumerate(self.test_sizes[size]) :
                inp[i,padding:-padding] = test_input[index]
                target[i,:] = test_target[index]
            self.test_input.append(inp)
            self.test_target.append(target)
        self.test_weights = np.sum(self.test_target,axis=1)
        self.test_weights[self.test_weights>0] = 1/self.test_weights[self.test_weights>0]/num_class
        
    def set_placeholders(self,pl_list):
        self.ph_x = pl_list[0]
        self.ph_y = pl_list[1]
        self.ph_w = pl_list[2]

    def get_next_batch(self):
        bi = self.batch_index
        self.batch_index += 1
        if self.batch_index == self.batch_size:
            self.batch_index = 0
            self.accumulate = False        
        return {
            self.ph_x: self.train_input[bi], 
            self.ph_y: self.train_target[bi], 
            self.ph_w: self.train_weights[bi]}
    
    def accumulate_grad(self):
        if self.accumulate :
            return True
        else :
            self.accumulate = True
            return False

    def get_next_dev_batch(self):
        bi = self.test_batch_index
        self.test_batch_index += 1
        if self.test_batch_index == self.test_batch_size:
            self.test_batch_index = 0
            self.accumulate_test = False        
        return {
            self.ph_x: self.test_input[bi], 
            self.ph_y: self.test_target[bi], 
            self.ph_w: self.test_weights[bi]}

    def accumulate_dev(self):
        if self.accumulate_test :
            return True
        else :
            self.accumulate_test = True
            return False
    
    def train_ended(self):
        pass