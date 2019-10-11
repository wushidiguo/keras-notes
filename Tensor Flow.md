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
### 1. 神经网络
1.1 顺序模型
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])