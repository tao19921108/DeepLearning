# 使用ImageDataGenerator读取图片

## 1. 创建ImageDataGenerator

```python
from tensorflow import keras

train_dir = './train'   # 训练数据集文件夹， 包含多个子文件夹，每个子文件夹包含一类图片
test_dir = './test'
# 创建ImageDataGenerator
train_data_gene = keras.preprocessing.image.ImageDataGenerator(
	rescale=1.0 / 255, # 像数值 * rescale
    rotation_range=40,  # 图片旋转角度范围
    width_shift_range=
)
```
