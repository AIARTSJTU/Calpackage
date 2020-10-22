# Calpackage
 (PYTORCH BASE)Calculation package for LPIPS, PSNR, SSIM_value
 
# Quick start
把这个文件夹放到你的工程根目录即可

Run `pip install lpips`. The following Python code is all you need.
```python
import lpips
import torch
from Calpackage.calpackage import calpackage
from torch.autograd import Variable
img1 = Variable(torch.rand(100, 3, 256, 256)) # image should be RGB, IMPORTANT: normalized to [-1,1]
img2 = Variable(torch.rand(100, 3, 256, 256)) # image should be RGB, IMPORTANT: normalized to [-1,1]
caltool = calpackage()
LPIPS_value_alexnetbase,LPIPS_value_vggnetbase,PSNR, SSIM = caltool.call(img1,img2)
print(LPIPS_value_alexnetbase.mean(),LPIPS_value_vggnetbase.mean(),PSNR, SSIM)
# example output：tensor(0.1543, grad_fn=<MeanBackward0>) tensor(0.3847, grad_fn=<MeanBackward0>) 13.801886697338164 tensor(0.0210, device='cuda:0')
print(LPIPS_value_alexnetbase.detach().numpy().mean(),LPIPS_value_vggnetbase.detach().numpy().mean(),PSNR, SSIM.cpu().detach().numpy())
# covert all these values to numpy type
#example output :0.15431352 0.38470986 13.801886697338164 0.021014081
