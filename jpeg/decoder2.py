import numpy as np
import simplejpeg

with open("images/tractor_color.jpeg", "rb") as f:
    a = simplejpeg.decode_jpeg(f.read())