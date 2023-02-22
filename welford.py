class WelfordVarianceEstimator:
    '''
        This class implements a version of Welford's algorithm 
        for calculating sample variance and mean for online streams
        of data
    '''
    def __init__(self, init_x, weight=1):
        '''
            Parameters:
                init_x - The initial array of data
        '''
        # Running mean
        self.M_k = init_x[0]
        self.S_k = 0
        self.k = 1
        for x in init_x[1:]:
            self.step(x)
            
    def step(self, x):
        self.k += 1
        M_k1 = self.M_k + (x-self.M_k)/self.k
        self.S_k = self.S_k + (x-self.M_k)*(x-M_k1)
        self.M_k = M_k1

    def get_variance(self):
        if self.k < 2:
            print("Cannot calculate variance of a two element list")
        return self.S_k/(self.k-1)
    
    def get_mean(self):
        return self.M_k