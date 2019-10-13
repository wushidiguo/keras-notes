## 一. 数据
### 1. 导入数据
#### 1.1 从链接
```python
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
```
#### 1.2 从keras数据集
```python
fashion_mnist = keras.datasets.fashion_mnist
```

#### 1.3 通过tfds.load()
```
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)
```

#### 1.4 pass

### 2. 数据处理
#### 2.1 数据清洗
```python
# 统计各列中为出现NaN的次数。isna = isnull
dataset.isna().sum()

# 将含NaN的数据做丢弃处理，默认行，可指定axis，非原地，可设置inplace。
dataset = dataset.dropna()
```
或
```python
# 将数据中的NaN置换为指定值。可通过method=指定填充或插值方式
dataset.fillna(0)
```
#### 2.2 train test分离
- 手动
```python
# 按照一定比例从数据集中分离出训练数据集。
train_dataset = dataset.sample(frac=0.8,random_state=0)
# 从元数据集中删除训练数据集作为测试数据集。
test_dataset = dataset.drop(train_dataset.index)
```
#### 2.3 One-Hot编码
- 手动
```python
origin = dataset.pop('Origin')  # pop方法可以将所选列从原数据块中弹出，原数据块不再保留该列。
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
```
- get_dummies
```python
dataset = pd.get_dummies(dataset)
```
#### 2.3 标准化
- 手动
```python
# 生成数据集的描述表。
train_stats = train_dataset.describe()
# 将描述表的行列倒置。
train_stats = train_stats.T
# 均值为0，标准差为1。
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']
```
## 二. 模型
### 1. 模型构建keras.Sequential() 
```python
# 通过将层的实例列表作为参数。
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

# 通过.add()方法。
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

# Sequential的第一层且只有第一层，需要接收输入的形状信息。注意，input_shape不包含batch大小，但可以通过batch_size另行指定。
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
# 等效于
model = Sequential()
model.add(Dense(32, input_dim=784))
```
keras提供了各种层，有时间慢慢读官方文档吧。[keras核心层](https://keras.io/layers/core/)

### 2. 模型编译model.compile() 
compilation，定义学习过程。通过compile方法，接收三个参数：
#### 2.1 优化器keras.optimizers
```python
# 传入优化器实例。
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
# 或传入优化器名称，使用默认参数。
model.compile(loss='mean_squared_error', optimizer='sgd')
```
keras提供了各种优化器。[keras优化器](https://keras.io/optimizers/)
#### 2.2 损失函数keras.losses  
损失函数接收两个参数y_true和y_pred。并为每一个数据点返回一个标量。
```python
# 传入函数。
model.compile(loss=losses.mean_squared_error, optimizer='sgd')
# 或传入函数名称。
model.compile(loss='mean_squared_error', optimizer='sgd')
```
- mean_squared_error 均方误差
- mean_absolute_error 平均绝对误差
- categorical_crossentropy 交叉熵（targets是one-hot编码）
- sparse_categorical_crossentropy （targets是数字编码）
- binary_crossentropy （一般用于二分类）  
...  
[keara损失函数](https://keras.io/zh/losses/)
#### 2.3 性能评估函数keras.metrics
用于评价模型表现的函数。对于分类问题，metrics=["accuracy"]。
```python
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
