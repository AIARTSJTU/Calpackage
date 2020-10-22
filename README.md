# Calpackage
 Calculation package for LPIPS, PSNR, SSIM_value
 
# Quick start
Run `pip install lpips`. The following Python code is all you need.
```python
import lpips
import torch
from calpackage import calpackage
from torch.autograd import Variable
img1 = Variable(torch.rand(100, 3, 256, 256))
img2 = Variable(torch.rand(100, 3, 256, 256))
caltool = calpackage()
LPIPS_value_alexnetbase,LPIPS_value_vggnetbase,PSNR, SSIM = caltool.call(img1,img2)
