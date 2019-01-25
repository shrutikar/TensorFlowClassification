import numpy as np


class AllData:
    def generate_data():
        data_tr = np.random.rand(1000,2)
        data_val = np.random.rand(100,2)
        l_tr = np.zeros([1000,3])
        l_val = np.zeros([100,3])

        def sample_class(sample) :
            if (sample[0]*sample[0]+sample[1]*sample[1])<0.9:
                return 0
            else :
                if sample[1]>0.5:
                    return 1
                else:
                    return 2

        for i in range(len(l_tr)) :
            l_tr[i][sample_class(data_tr[i])]=1
        for i in range(len(l_val)) :
            l_val[i][sample_class(data_val[i])]=1
        return data_tr,l_tr,data_val,l_val
            
    def __init__(self, train_input, train_target, test_input, test_target):
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target
        self.counter = False
        
    def set_placeholders(self,pl_list):
        self.ph_x = pl_list[0]
        self.ph_y = pl_list[1]

    def get_next_batch(self):
        return {self.ph_x: self.train_input, self.ph_y: self.train_target }
    
    def accumulate_grad(self):
        self.counter = not self.counter
        return self.counter

    def get_next_dev_batch(self):
        return {self.ph_x: self.test_input, self.ph_y: self.test_target }
    
    def train_ended(self):
        pass

