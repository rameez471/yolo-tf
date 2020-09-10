"""Miscellaneous utility functions"""
from functools import reduce
import tensorflow as tf
from PIL import Image

def compose(*func):
    """Compose functions"""
    if func:
        return reduce(lambda f,g: lambda *a,**kw:g(f(*a,**kw)),func)
    else:
        raise ValueError('Composition of empty function not supported')

def letter_box(image,size):
    '''Resize image with unchanges aspect ratio using padding'''
    image_width, image_height = image.size
    width, height = size
    scale = min(width/image_width,height/image_height)
    new_width = int(image_width * scale)
    new_height = int(image_height * scale)

    image = image.resize((new_width,new_height),Image.BICUBIC)
    new_image = Image.new('RGB',size,(128,128,128))
    new_image.paste(image,((width-new_width)//2,(height-new_height)//2))

    return new image