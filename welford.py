class WelfordVarianceEstimator:
    def __init__(self, init_x):
        self.M_k = init_x[0]
        self.S_k = 0
        self.k = 1
        for x in init_x[1:]:
            self.step(x)
            
    def step(self, x):
        M_k1 = self.M_k + (x-self.M_k)/self.k
        self.S_k = self.S_k + (x-self.M_k)*(x-M_k1)
        self.M_k = M_k1
        self.k += 1

    def get_variance(self):
        return self.S_k/(self.k-1)