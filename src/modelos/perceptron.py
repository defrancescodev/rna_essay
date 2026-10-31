import numpy as np

class Perceptron:
    def __init__(self):
        self.eta = 0.0001
        self.max_epochs = 100  
        self.w = None
        self.epoca = 0
        self.history_w = []

    @staticmethod
    def degrau_bipolar(u_k):
        return 1 if u_k >= 0 else -1

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        return np.vstack((-np.ones((1, N)), X.T))

    def fit(self, X: np.ndarray, d: np.ndarray):
        N, p = X.shape
        X_biased = self.add_bias(X)
        
        self.w = np.random.random_sample((p + 1, 1)) - 0.5
        self.history_w.append(self.w.copy())
        
        erro = True
        self.epoca = 0

        while erro and self.epoca < self.max_epochs:
            erro = False
            for k in range(N):
                x_k = X_biased[:, k].reshape(p + 1, 1)
                d_k = d[k, 0]
                
                u_k = (self.w.T @ x_k)[0, 0]
                y_k = self.degrau_bipolar(u_k)
                e_k = d_k - y_k
                
                self.w = self.w + self.eta * e_k * x_k
                if e_k != 0:
                    erro = True
                    
            self.epoca += 1
            self.history_w.append(self.w.copy())