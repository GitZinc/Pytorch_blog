# Part4 神经网络工具箱
autograd虽然可以实现深度学习模型，但是抽象程度低，编写代码量大。这种情况下,torch.nn应运而生。

torch.nn的核心数据结构是**Module**,这是一个抽象概念，既可以表示神经网络中的某个层，也可以表示含多个层的神经网络。

最常见的做法是，继承nn.Module，撰写自己的网络/层。下面利用nn.Module实现自己的全连接层。
输入y与输出x满足$$y=Wx+b$$
可见，全连接层的实现非常简单，其代码量不超过10行，但需注意以下几点：

- 自定义层Linear必须继承nn.Module，并且在其构造函数中需调用nn.Module的构造函数，即super(Linear, self).__init__() 或nn.Module.__init__(self)，推荐使用第一种用法，尽管第二种写法更直观。
- 在构造函数__init__中必须自己定义可学习的参数，并封装成Parameter，如在本例中我们把w和b封装成parameter。parameter是一种特殊的Variable，但其默认需要求导（requires_grad = True），感兴趣的读者可以通过nn.Parameter??，查看Parameter类的源代码。
- forward函数实现前向传播过程，其输入可以是一个或多个variable，对x的任何操作也必须是variable支持的操作。
- 无需写反向传播函数，因其前向传播都是对variable进行操作，nn.Module能够利用autograd自动实现反向传播，这点比Function简单许多。
- 使用时，直观上可将layer看成数学概念中的函数，调用layer(input)即可得到input对应的结果。它等价于layers.__call__(input)，在__call__函数中，主要调用的是 layer.forward(x)，另外还对钩子做了一些处理。所以在实际使用中应尽量使用layer(x)而不是使用layer.forward(x)，关于钩子技术将在下文讲解。
- Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器，前者会给每个parameter都附上名字，使其更具有辨识度。
可见利用Module实现的全连接层，比利用Function实现的更为简单，因其不再需要写反向传播函数。
 
## 多层感知机
![多层感知机](http://localhost:8888/files/imgs/multi_perceptron.png)

即使是稍微复杂一些的多层感知机，实现也并不复杂。
- 构造函数__init__中,可利用自定义的Linear层(module)，作为当前module对象的一个子module，它的可学习参数，都会成为当前module的可学习参数。
 
 
 ## 常用神经网络层
 ### 图像相关层
 卷积，池化层。
 
 #### 4.1.2 激活函数
PyTorch实现了常见的激活函数，其具体的接口信息可参见官方文档[^3]，这些激活函数可作为独立的layer使用。这里将介绍最常用的激活函数ReLU，其数学表达式为：
$$ReLU(x)=max(0,x)$$
[^3]: http://pytorch.org/docs/nn.html#non-linear-activations
```
relu = nn.ReLU(inplace=True)
input = V(t.randn(2, 3))
print(input)
output = relu(input)
print(output) # 小于0的都被截断为0
# 等价于input.clamp(min=0)
```
```
Variable containing:
 0.4437 -1.2088 -0.6859
-1.9389 -0.5563 -1.5315
[torch.FloatTensor of size 2x3]

Variable containing:
 0.4437  0.0000  0.0000
 0.0000  0.0000  0.0000
[torch.FloatTensor of size 2x3]
```
以上的例子中基本都是将每一层的输出输入到下一层中,这样的网络成为前馈神经网络.对于此类网络每次都写forward函数会有些麻烦,所以有两种简化方式,ModuleList和Sequential.Sequential是一个特殊的,Module,包含几个子module,先前传播的时候会将输入层传递下去.ModuleList也是特殊的module,包含几个子module,可以像List一样使用,但不能直接把输入传递到ModuleLis.

# Sequential的三种写法
```
net1 = nn.Sequential()
net1.add_module('conv',nn.Conv2d(3,3,3))
net1.add_module('batchnorm',nn.BatchNorm2d(3))
net1.add_moudle('activation_layer', nn.ReLU())

net2 = nn.Sequential(
# Conv2d是二维卷积,第一个参数为输入通道数,第二个为输出通道数
		nn.Conv2d(3,3,3),
        nn.BatchNorm2d(3),
        nn.ReLU()
        )

from collections import OrderedDict
net3 = nn.Sequential(OrderedDict([
		('Conv1',nn.Conv2d(3,3,3)),
        ('bn1',BatchNorm2d(3)),
        ('relu1', nn.ReLU())
        ]))
print('net1:',net1)
print('net2:',net2)
print('net3:',net3)
```

# 可根据名字或序号取出子module
```
input:
net1.conv, net2[0], net3.conv1
output:
(Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
 Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)),
 Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)))
 ```
> 你太累，该摸了。
> ——Mona
