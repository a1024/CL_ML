import torch

x = torch.ones(5, 5, requires_grad=True)
print(x)

# temporarily set the first row of x to zero
with torch.no_grad():
    x[0, :] = 0

print(x)
