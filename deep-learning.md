## 一、 优化器（optimizer）
### 1. 随机梯度下降法（SGD）
#### 1.1 SGD的实现

```python
class SGD:
    # 初始化参数lr表示学习率。
    def __init__(self, lr=0.01):
        self.lr = lr
        
    # params和grads为字典对象，分别保存权重参数和梯度。
    def update(self, params, grads):
        for key in params.keys():
            params[keys] -= self.lr * grad[key]
```
#### 1.2 SGD的缺点
- 如果函数的形状非均向，比如呈延伸状，搜索的路径就会非常低效。SGD低效的根本原因是，梯度的方向并没有指向最小值的方向（之字路线）。
![SGD更新路径](https://user-gold-cdn.xitu.io/2019/10/3/16d8f6d0256d4eaa?w=664&h=464&f=png&s=50907)

### 2. Momentum
#### 2.1 Momentum的实现

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum =momentum
        self.v = None
        
    def update(self, params, grads):
        # v以字典型变量的形式保存于参数结构相同的数据，初始值全部为零。
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```
#### 2.2 Momentum的优点
Momentum的原理如下：
$$\nu\leftarrow\alpha\nu\ -\ \eta\frac{\partial L}{\partial W}$$
$$W\leftarrow W + \nu$$
更新路径就像小球在碗中滚动。v是速度，$- \eta \frac{\partial L}{\partial W}$代表加速度。相较于SGD，Momentum可以更快地向收敛点靠近，减弱“之字形”的变动程度。

![Momentum更新路径](https://user-gold-cdn.xitu.io/2019/10/3/16d8f6e35c677812?w=660&h=445&f=png&s=53302)
### 3. AdaGrad

#### 3.1 AdaGrad的实现

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h =None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.item():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            # 加上微小值1e-7是为了放置self.h[key]中有0。
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```
#### 3.2 AdaGrad的优点
- AdaGrad的原理如下：
$$h\leftarrow h + \frac{\partial L}{\partial W} \odot \frac{\partial L}{\partial W}$$
$$W \leftarrow W-\eta \frac{1}{\sqrt h} \frac{\partial L}{\partial W}$$
AdaGrad作为**学习率衰减**方法的一种，增加了变量**h**，它保存以前所有梯度值的平方和，在更新参数时，变动较大（被大幅更新)的元素学习率将变小。

![AdaGrad更新路径](https://user-gold-cdn.xitu.io/2019/10/3/16d8f6f1ca135868?w=659&h=466&f=png&s=54037)
### 4. Adam
Adam融合了Momentum和AdaGrad方法。能实现参数空间的高效搜索。进行超参数的“偏置校正”也是Adam的特征。

![Adam更新路径](https://user-gold-cdn.xitu.io/2019/10/3/16d8f729b388f64c?w=659&h=464&f=png&s=54048)
### 5.优化器的选择
基于MNIST数据集的4种更新方法的比较：

![四种更新方法的比较](https://user-gold-cdn.xitu.io/2019/10/3/16d8fa6e89362447?w=663&h=362&f=png&s=65546)
虽然实验结果会随着学习率等超参数、神经网络结构的不同而变化。但一般而言，与SGD相比，其他3种方法学习得更快，有时最终的识别精度也更高。

## 二、权重初始化
#### 要点
- 通过减小权重参数的值可以抑制过拟合的发生。因此一般将初始权重设置为较小的值，如标准差为0.01的高斯分布：`0.01 * np.random.randn(10, 100)`。
- 如果将权重初始值设成一样的值，将无法正确进行学习。这是因为所有权重将进行完全相同的更新。
- 希望各层的激活值（激活函数的输出数据）的分布有适当的广度，以避免出现梯度消失或“表现力受限”的问题。一般采用Xavier初始值，即前一层的节点数为n，初始值使用标准差为$\frac{1}{\sqrt n}$的分布。

```python
node_num = 100 # 前一层的节点数
W = np.random.randn(node_num, node_num) / np.sqrt(node_num)
```
- Xavier适合于sigmoid函数和tanh函数。当激活函数使用ReLU时，推荐使用“He初始值”。即前一层的节点数为n，初始值使用标准差为$\frac{2}{\sqrt n}$的分布。
- 在神经网络的学习中，权重初始值非常重要/如下图所示，有时使用错误的初始值，完全无法进行学习。
![基于MNIST数据集的权重初始值的比较](https://user-gold-cdn.xitu.io/2019/10/3/16d8fff9126eafd3?w=660&h=456&f=png&s=59692)

## 三、 超参数
#### 超参数
神经网络中，各层的神经元数量，batch大小，学习率或权值衰减等，称为超参数。
#### 注意
不能用测试数据评估超参数的性能。调整超参数时，必须使用专用数据，即验证数据（validation data）。<br/>
训练数据————>参数（权重和偏置）的学习；<br/>
验证数据————>超参数的性能评估；<br/>
测试数据————>确认泛化能力。<br/>
#### 最优化
##### 步骤0
设定超参数的范围（对数尺度），如0.001到1000。
##### 步骤1
从设定范围中随机采样。
```python
# 随即均匀采样
10 ** np.random.uniform(-3, 3)
```
##### 步骤2
使用采样到的超参数进行学习，通过验证数据评估识别精度（要将epoch设置得很小）。
##### 步骤3
重复步骤1和2（100次等），根据识别精度的结果，缩小超参数范围。重复上述步骤，直至范围缩小到一定程度，从中选出一个超参数的值。

## 四、 正则化
#### 发生过拟合的原因
- 模型拥有大量参数，表现力强。
- 训练数据少。
#### 权值衰减
该方法通过在学习过程中对大的权重进行惩罚，来抑制过拟合。**很多过拟合原本就是因为权重参数取值过大才发生的。**
<br/>
具体方式为，为损失函数加上权重的范数，例如平方范数（L2范数）。将权重记为**W**，L2范数的权值衰减就是$\frac{1}{2}\lambda W^2$，将此项加到损失函数上。λ是控制正则化强度的超参数。lambda设置得越大，对大的权重施加的惩罚就越重。对于所有权重，权值衰减方法都会为损失函数加上是$\frac{1}{2}\lambda W^2$。
#### Dropout
Dropout是一种在学习的过程中随机删除（隐藏层）神经元的方法。被删除的神经元不再进行信号传递。如图所示：

![Dropout概念图](https://user-gold-cdn.xitu.io/2019/10/3/16d90a8e70428287?w=663&h=296&f=png&s=109454)  

测试时，会传递所有的神经元信号，各个神经元的输出，要乘上训练时的删除比例后再输出。可以理解为，每一次让不同的模型进行学习，推理时区模型的平均值。

```python
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
    
    def forward(self, x, train_flg=True):
        # 学习过程train_flg为真，随机生成x形状的mask([0, 1)区间，均匀分布)，将值比dropout_ratio大的元素设为True。
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        # 测试时train_flg为假，x要乘上训练时的非零比例后输出。
        else:
            return x * (1.0 - self.dropout_ratio)
    # 反向传播时，类似于ReLU。
    def backward(self, dout):
        return dout * self.mask
```
## 五、 Batch Normalization
#### Batch Normalization的优点
- 可以加快学习进度。
- 降低对初始值的依赖。
- 抑制过拟合。
#### Batch Norm层
通过向神经网络中插入Batch Norm层，即以学习时的mini-batch为单位进行正则化，使数据分布满足均值为0，方差为1。

![插入Batch Norm层的神经网络](https://user-gold-cdn.xitu.io/2019/10/3/16d90161951afad8?w=663&h=204&f=png&s=15240)
$$\mu _B \leftarrow \frac{1}{m} \sum_{i=1}^m x _i$$
$$\sigma _B ^2 \leftarrow \frac{1}{m} \sum_{i=1}^m (x _i - \mu _B) ^2$$
$$\hat x _i \leftarrow \frac{x _i - \mu _B}{\sqrt {\sigma _B^2+\varepsilon}}$$
$\varepsilon$是一个微小值（如，10e-7），是为了防止除以0的情况。
接着对正规化后的数据进行缩放和平移变换。
$$y _i \leftarrow \gamma \hat x_i + \beta$$
一开始，$\gamma = 1$, $\beta = 0$,然后在通过学习调整到合适的值。

## 六、 卷积神经网络CNN
#### 概念
- **全连接层（Affine）**：在全连接层中，相邻层的神经元全部连接在一起，输出的数量可以任意决定。<br/>
全连接层的问题在与忽视了数据的形状，对于图像这样的3维数据，需拉平为1维数据进行处理。
- **卷积**：卷积运算相当于图像处理中的“滤波器运算”。以一定间隔滑动滤波器的窗口，将各个位置上滤波器的元素和对应元素相乘，然后求和并保存到输出的对应位置。
![卷积运算](https://user-gold-cdn.xitu.io/2019/10/4/16d94a223dea08ef?w=535&h=299&f=webp&s=87392)
- **填充（padding)**：向输入数据的周围填入固定的数据（比如0等），称为填充。其目的是为了调整输出的大小。
![padding](https://user-gold-cdn.xitu.io/2019/10/4/16d948d29c397945?w=530&h=225&f=png&s=35532)
- **步幅（stride）**：应用滤波器的位置间隔称为步幅。
![步幅为2的卷积](https://user-gold-cdn.xitu.io/2019/10/4/16d949104ed99ff9?w=530&h=379&f=png&s=50866)  

对于输入(H, W)，滤波器(FH, FW)，输出(OH, OW)，填充P，步幅S。则
$$OH=\frac{H+2P-FH}{S}+1$$
$$OW=\frac{W+2P-FW}{S}+1$$
- **3维数据卷积**：
在3维数据的卷积运算中，滤波器的通道数要和输入数据的通道数保持一致。按通道进行输入数据和滤波器的卷积运算，并将结果相加，从而得到输出。

![3维数据卷积运算](https://user-gold-cdn.xitu.io/2019/10/4/16d958409790467e?w=531&h=213&f=png&s=33940)  

通过应用FN个滤波器，可以得到具有FN个通道的输出。此时滤波器的全中数据要按照(output_channel, input_channel, height, width)的顺序书写。偏置的形状时(FN, 1, 1)。完整的处理流如下：

![卷积运算的处理流](https://user-gold-cdn.xitu.io/2019/10/4/16d9596216c1704b?w=827&h=284&f=png&s=58831)
- **池化层**：池化是缩小高、宽方向上的空间的运算。Max池化是从目标区域取出最大值，Average池化则是计算目标区域的平均值。在图像识别领域，主要使用Max池化。池化运算按通道独立进行。
#### 卷积层和池化层的实现
- **im2col**
im2col是一个函数，将输入数据展开以适合滤波器（权重），将批量的卷积运算归结到矩阵运算，以充分利用线性代数库。数据的展开过程如图所示：

![im2col数据展开](https://user-gold-cdn.xitu.io/2019/10/5/16d9982586ede147?w=1192&h=619&f=png&s=78406)

![卷积核展开](https://user-gold-cdn.xitu.io/2019/10/5/16d99833cafc5c6d?w=635&h=397&f=png&s=30419)
具体实现请参考[此文](https://blog.csdn.net/Daycym/article/details/83826222)。

- **卷积层的实现**
```python
import numpy as np

def im2col(input_data, dilter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 卷积核的高
    filter_w : 卷积核的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    # 输入数据的形状
    # N：批数目，C：通道数，H：输入数据高，W：输入数据长
    N, C, H, W = input_data.shape  
    out_h = (H + 2*pad - filter_h)//stride + 1  # 输出数据的高
    out_w = (W + 2*pad - filter_w)//stride + 1  # 输出数据的长
    # 填充 H,W
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    # (N, C, filter_h, filter_w, out_h, out_w)的0矩阵
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
```

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        # 卷积核的形状
        FN, C, FH, FW = self.W.shape
        # 输入的形状
        N, C, H, W = x.shape
        # 计算输出特征图的形状
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 展开滤波器
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        
        # (N, H, W, C)——>(N, C, H, W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        return out
```
注意，卷积层的输出深度C是一个超参数，它与使用的滤波器数量一致。每种滤波器所做的就是在输入数据中找寻一种特征。
- **池化层的实现**
池化层的实现与卷积层相同，都使用im2col展开输入数据。但池化的情况下，在通道方向上是独立的。

![池化层的实现流程](https://user-gold-cdn.xitu.io/2019/10/5/16d9a43673a7adda?w=1047&h=527&f=png&s=114837)

```python
class Pooling:
    # pool_h, pool_w为池化半径
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    
    def forward(self, x):
        N, C, H, W = x.shape
        # 池化层的输出尺寸
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        
        # 展开
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool*self.pool_w)
        
        # 最大值
        out = np.max(col, axis=1)
        # 转化
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        return out
```
- CNN的实现
实现如图所示的CNN：

![CNN网络构成](https://user-gold-cdn.xitu.io/2019/10/5/16d9aa7e0aafa293?w=1052&h=351&f=png&s=26450)
```python 
class simpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_param={"filter_num":30, "filter_size":5, "pad":0, "stride":1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        # 卷积核数
        filter_num = conv_param["filter_num"]
        # 卷积核尺寸
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["filter_pad"]
        filter_stride = conv_param["filter_stride"]
        # 输入特征图尺寸
        input_size = input_dim[1]
        # 卷积层输出特征图尺寸
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        # 池化层输出神经元数量。特征图各边缩小为原来的二分之一。
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        
        # 权重参数初始化
        # weight_init_randn初始化权重的标准差
        self.params = {}
        # 卷积层权重和偏置
        self.params["W1"] = weight_init_randn * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        # 第一全连接层权重和偏置
        self.params["W2"] = weight_init_randn * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        # 第二全连接层权重和偏置
        self.params["W3"] = weight_init_randn * np.random.randn(hidden_size,output_size)
        self.params["b3"] = np.zeros(output_size)
        
        # 添加卷积层
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], conv_param["stride"], conv_param["pad"])
        # 添加第一个ReLU层
        self.layers["Relu1"] = Relu()
        # 添加一个2x2、步幅为2的池化层
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        # 添加第一个全连接层
        self.layers["Affine1"] = Affine(self.params["W2]", self.params["b2"])
        # 添加第二个ReLU层
        self.layers["Relu2"] = Relu()
        # 添加第二个全连接层
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        # 将softmaxwithloss层独立保存，因为此层的参数包含y和t。
        self.last_layer = softmaxwithloss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
        
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    # 误差反向传播
        def gradient(self, x, t):
            # forward
            self.loss(x, t)
            
            # backward
            dout = 1
            dout = self.lastlayer.backward(dout)
            
            layers = list(self.layers.values())
            # 调转layers顺序
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)
            
            # 取出梯度
            grads = {}
            grads["W1"] = self.layers["Conv1"].dW
            grads["b1"] = self.layers["Conv1"].db
            grads["W2"] = self.layers["Affine1"].dW
            grads["b1"] = self.layers["Affine1"].db
            grads["W3"] = self.layers["Affine2"].dW
            grads["b3"] = self.layers["Affine2"].db
            
            return grads
```
 此处似乎有错误。在pooling层之后，少了flatten的步骤，而直接与affine层连接。下图是比较正确的连接方式。
 
![卷积神经网络](https://user-gold-cdn.xitu.io/2019/10/6/16d9eefb96df00c7?w=1644&h=880&f=jpeg&s=216825)
现在的CNN中主要使用ReLU作为激活函数。随着CNN层次加深，提取的信息也愈加复杂、抽象。叠加小型滤波器来加深网络的好处是可以减少参数的数量，扩大感受野（receptive field， 给神经元施加变化的某个局部空间区域）。并且，通过叠加层，将ReLU等激活函数夹在卷积层的中间，进一步提高了网络的表现力。