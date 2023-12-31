import numpy as np

class EarlyStopping:
    def __init__(self, patience:int = 3):
        self.patience = patience
        self.counter = 0
        self.best_loss:float = np.inf
        self.best_model_pth = 0

    def __call__(self, loss, epoch: int):
        should_stop = False

        if loss >= self.best_loss:
            self.counter += 1
            if self.counter > self.patience:
                should_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
            self.best_model_pth = epoch
        return should_stop