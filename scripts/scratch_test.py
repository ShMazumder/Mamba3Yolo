import torch
import torch.nn as nn

class MambaTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.dt_proj = nn.Linear(32, 32)
        
    def forward(self, x):
        uf = x.float()
        return self.dt_proj(uf)

m = MambaTest().cuda().half()  # like validator
x = torch.randn(2, 32).cuda().half()
try:
    m(x)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
