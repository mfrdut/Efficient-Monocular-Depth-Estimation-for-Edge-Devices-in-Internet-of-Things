import torch
import math
import numpy as np

def log10(x):
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        
    def set_to_worst(self):
        #irmse, imae,  
        self.irmse, self.imae = np.inf, np.inf
        #rel, delta1
        self.absrel, self.lg10 = np.inf, np.inf
        #delta2, delta3
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        #mse, rmse,  mae
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3):
        # update 
        self.irmse, self.imae = irmse, imae   
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.mse, self.rmse, self.mae = mse, rmse, mae

    def evaluate(self, pred, target):
        valid = target>0
        target = target[valid]
        pred = pred[valid]
        diff_abs = (pred - target).abs()
         
        self.mse = (torch.pow(diff_abs, 2)).mean()
        self.rmse = math.sqrt(self.mse)
        self.mae = diff_abs.mean()
        self.absrel = (diff_abs / target).mean()
        self.lg10 = (log10(pred) - log10(target)).abs().mean()

        Ratio_max = torch.max(pred / target, target / pred)
        self.delta1 = (Ratio_max < 1.25).float().mean()
        self.delta2 = (Ratio_max < 1.25 ** 2).float().mean()
        self.delta3 = (Ratio_max < 1.25 ** 3).float().mean()
        
        inv_pred = 1 / pred
        inv_target = 1 / target
        inv_diff_abs = (inv_pred - inv_target).abs()
        self.imae = inv_diff_abs.mean()
        self.irmse = math.sqrt((torch.pow(inv_diff_abs, 2)).mean())

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.number = 0.0

        self.mse_sum, self.rmse_sum, self.mae_sum = 0, 0, 0
        self.irmse_sum, self.imae_sum = 0, 0
        self.delta1_sum, self.delta2_sum, self.delta3_sum = 0, 0, 0
        self.absrel_sum, self.lg10_sum = 0, 0
    #sum
    def update(self, result, n=1):
        self.number += n

        self.imae_sum += n*result.imae
        self.irmse_sum += n*result.irmse        
        self.mse_sum += n*result.mse
        self.rmse_sum += n*result.rmse
        self.absrel_sum += n*result.absrel
        self.mae_sum += n*result.mae
        #sum: delta1, delta2, delta3
        self.delta1_sum += n*result.delta1
        self.delta2_sum += n*result.delta2
        self.delta3_sum += n*result.delta3
        self.lg10_sum += n*result.lg10

    #average
    def average(self):
        avg = Result()
        avg.update(
            self.irmse_sum / self.number, self.imae_sum / self.number,
            self.mse_sum / self.number, self.rmse_sum / self.number, self.mae_sum / self.number, 
            self.absrel_sum / self.number, self.lg10_sum / self.number,
            self.delta1_sum / self.number, self.delta2_sum / self.number, self.delta3_sum / self.number
            )
        return avg