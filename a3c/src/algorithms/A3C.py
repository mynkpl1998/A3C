import sys
from copy import deepcopy
from a3c.src.algoConfigs.A3C import A3C_DEFAULT_DICT

class A3C:

    def __init__(self, ):
        self.config = deepcopy(A3C_DEFAULT_DICT)
