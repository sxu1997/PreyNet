import torch
a = torch.randn(4,1,10,10)
b = torch.randn(4,1,10,10)
score_a = torch.sum((a*b)) / torch.sum(b)
y = 0.0
for i in range(4):
    y += torch.sum(a[i]*b[i]) / torch.sum(b[i])
y = y/(4)
print(score_a)
print(y)



