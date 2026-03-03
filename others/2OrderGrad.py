import torch  # 以PyTorch为例

# 定义一个可微的变量x
x = torch.tensor([1.0], requires_grad=True)

# 定义函数f(x)
f = x ** 2 + 3 * x + 2

# 计算f关于x的一阶导数
f.backward()

# x的梯度现在包含f关于x的一阶导数
grad1 = x.grad

# 重置x的梯度，准备计算二阶导数
x.grad = None

# 计算二阶导数
grad1.backward()

# x的梯度现在包含f关于x的二阶导数
grad2 = x.grad

print(grad2)
