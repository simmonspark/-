import torch

a = torch.Tensor([0]).to(dtype=torch.float16)
b = torch.Tensor([4]).to(dtype=torch.int8)
a + b

#파이토치는 되노
print(a+b)


