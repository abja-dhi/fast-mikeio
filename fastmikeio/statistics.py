import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dfsu import Dfsu3DSigma, DfsuVerticalProfileSigma

class Statistics:
    def __init__(self, dfsu):
        self.dfsu = dfsu

    def quantile(self, q, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        assert 0 <= q <= 1, "Quantile q must be between 0 and 1."
        out = np.quantile(data, q, axis=0)
        return out
    
    def max(self, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        out = np.max(data, axis=0)
        return out
        
    def min(self, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        out = np.min(data, axis=0)
        return out

    def mean(self, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        out = np.mean(data, axis=0)
        return out


class Statistics3DSigma(Statistics):
    def __init__(self, dfsu: 'Dfsu3DSigma'):
        super().__init__(dfsu)



class StatisticsVerticalProfileSigma(Statistics):
    def __init__(self, dfsu: 'DfsuVerticalProfileSigma'):
        super().__init__(dfsu)