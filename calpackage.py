import pytorch_ssim
import torch
from torch.autograd import Variable
from imageReader import *
from psnr import *
import lpips


class calpackage(object):
    def __init__(self, mode = 'all', width = 400, height = 400):
        self.mode = mode
        self.width = width
        self.height = height
        self.npix = width * height

    def call(self, img1, img2):
        #lpips part
        loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
        loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
        lpips_value_alex = loss_fn_alex(img1, img2)
        lpips_value_vgg = loss_fn_vgg(img1, img2)

        # img1_01 = (img1.numpy()+1.)/2.
        # img2_01 = (img2.numpy()+1.)/2.

        #psnr part
        psnr = psnr_cal((img1.numpy()+1.)/2.,(img2.numpy()+1.)/2.)

        #ssim part
        if torch.cuda.is_available():
            img1 = img1.cuda()
            img2 = img2.cuda()

        #print(pytorch_ssim.ssim(img1, img2))

        ssim_loss = pytorch_ssim.SSIM(window_size = 11)
        ssim_value = ssim_loss(img1, img2)

        return lpips_value_alex,lpips_value_vgg, psnr, ssim_value

