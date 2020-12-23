import numpy as np
class Alg_ML:
    def __init__(self,data,params,max_iters=200):
        self.data = data
        self.params = params
        self.max_iters = max_iters
    def main(self,init,update,termin):
        self.var = init(self.data,self.params)
        for i in range(self.max_iters):
            self.var=update(self.data,self.var)

