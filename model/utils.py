"""Miscellaneous utility functions"""
from functools import reduce
import tensorflow as tf

def compose(*func):
    """Compose functions"""
    if func:
        return reduce(lambda f,g: lambda *a,**kw:g(f(*a,**kw)),func)
    else:
        raise ValueError('Composition of empty function not supported')
