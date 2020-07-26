#проверяем, что фичи в нормальном диапазоне и логируем

from typing import Callable
from functools import wraps
import logging



def check_invariants(func:Callable) -> Callable:
    @wraps(func)
    def invariants_checker(*args):
        print(1)
        _check_features_invariants(*args)
        return func(*args)
    return invariants_checker

def _check_features_invariants(x):
    print(2)
    #logging.basicConfig(filename='example2.log',level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.basicConfig(filename='example3.log', format='%(asctime)s %(message)s')
    
    if not x['key1'] > 1 and x['key1'] <=2: 
        logging.warning('is when this event was logged.')
        logging.debug(x['key1'] > 1 and x['key1'] <=2)
    #logger
    
'''
def _check_prediction_invariant(result, mode):
    pass
    print(result, mode)
    #throw an exception if mode is DEV
    #log if mode is PROD
'''

@check_invariants
def get_prediction(x:dict) -> float:
    print(3)
    return x['key1'] * 2 - 1

q = get_prediction({'key1':2.0001})
print(q)
