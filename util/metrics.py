import torch.nn as nn

# Refrence Link : https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/loss.py
class Jaccard_Metric(nn.Module):
    def __init__(self):
        super(Jaccard_Metric, self).__init__()
        self.smooth = 1

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (intersection + self.smooth) / (y_pred.sum() + y_true.sum() - intersection + self.smooth)
        return dsc
    
    
    
class Dice_Coefficient(nn.Module):
    def __init__(self):
        super(Dice_Coefficient, self).__init__()
        self.smooth = 1

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2 * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth)
        return dsc
    