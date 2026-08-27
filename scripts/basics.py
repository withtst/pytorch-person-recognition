import torch

x = torch.tensor([1.0, 2.0, 3.0])
print('x:', x)
print('x.dtype:', x.dtype)
print('x.device:', x.device)
print('x.shape:', x.shape)

print('x+1:', x+1)
print('x.sum():', x.sum())

print('torch.cuda.is_available():', torch.cuda.is_available())
if torch.cuda.is_available():
    x = x.to('cuda')
    print('after .to("cuda") x.device:', x.device)
    print('cuda device name:', torch.cuda.get_device_name(0))

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(4.0, requires_grad=True)
c = a * b
c.backward()
print('a.grad:', a.grad)
print('b.grad:', b.grad)

x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = x * 2
y = y.sum()
y.backward()
print('x.grad:', x.grad)