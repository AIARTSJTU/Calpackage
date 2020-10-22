import numpy
import math

def psnr_cal(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2, keepdims = False)
    # if mse == 0:
    #     return 100
    # PIXEL_MAX = 255.0
    # return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return -10.0 * math.log10(mse)
