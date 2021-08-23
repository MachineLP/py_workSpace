#导入相应的库
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import efficientnet.tfkeras as efn
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np  
import itertools
import numpy as np
import os

'''
#keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#keras.applications.resnet.ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#keras.applications.resnet.ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#keras.applications.resnet_v2.ResNet101V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#keras.applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)


from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# 构建不带分类器的预训练模型
base_model = ResNet101(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器，假设我们有200个类
predictions = Dense(200, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 在新的数据集上训练几代
model.fit_generator(...)

# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。

# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 我们选择训练最上面的两个 Inception block
# 也就是说锁住前面249层，然后放开之后的层。
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 我们需要重新编译模型，才能使上面的修改生效
# 让我们设置一个很低的学习率，使用 SGD 来微调
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 我们继续训练模型，这次我们训练最后两个 Inception block
# 和两个全连接层
model.fit_generator(...)

'''

#设置图片的高和宽，一次训练所选取的样本数，迭代次数
im_height = 224
im_width = 224
batch_size = 32
epochs = 15

# 创建保存模型的文件夹
if not os.path.exists("save_weights"):
    os.makedirs("save_weights")


image_path = "../input/10-monkey-species/" # 猫狗数据集路径
# 一个类别放到一个文件中
train_dir = image_path + "training/training" #训练集路径
validation_dir = image_path + "validation/validation" #验证集路径

# 定义训练集图像生成器，并进行图像增强
train_image_generator = ImageDataGenerator( rescale=1./255, # 归一化
                                            rotation_range=40, #旋转范围
                                            width_shift_range=0.2, #水平平移范围
                                            height_shift_range=0.2, #垂直平移范围
                                           shear_range=0.2, #剪切变换的程度
                                            zoom_range=0.2, #剪切变换的程度
                                            horizontal_flip=True,  #水平翻转
                                            fill_mode='nearest')
                                            
# 使用图像生成器从文件夹train_dir中读取样本，对标签进行one-hot编码
train_data_gen = train_image_generator.flow_from_directory(directory=train_dir, #从训练集路径读取图片
                                                           batch_size=batch_size, #一次训练所选取的样本数
                                                           shuffle=True, #打乱标签
                                                           target_size=(im_height, im_width), #图片resize到224x224大小
                                                           class_mode='categorical') #one-hot编码
                                                           
# 训练集样本数        
total_train = train_data_gen.n 

# 定义验证集图像生成器，并对图像进行预处理
validation_image_generator = ImageDataGenerator(rescale=1./255) # 归一化

# 使用图像生成器从验证集validation_dir中读取样本
val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,#从验证集路径读取图片
                                                              batch_size=batch_size, #一次训练所选取的样本数
                                                              shuffle=False,  #不打乱标签
                                                              target_size=(im_height, im_width), #图片resize到224x224大小
                                                              class_mode='categorical') #one-hot编码
                                                              
# 验证集样本数      
total_val = val_data_gen.n

'''
# 构建不带分类器的预训练模型
base_model = ResNet101(weights='imagenet', include_top=False)

#构建模型  
model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), #加入全局平均池化层
        tf.keras.layers.Dense(10, activation='softmax') #添加输出层(10分类)
    ])

'''

#使用efficientnet.tfkeras的EfficientNetB0网络，并且使用官方的预训练模型
covn_base = efn.EfficientNetB0(weights='imagenet', include_top=False ,input_shape=[im_height,im_width,3])
covn_base.trainable = True

#冻结前面的层，训练最后10层
for layers in covn_base.layers[:-10]:
    layers.trainable = False
    
#构建模型  
model = tf.keras.Sequential([
        covn_base,
        tf.keras.layers.GlobalAveragePooling2D(), #加入全局平均池化层
        tf.keras.layers.Dense(10, activation='softmax') #添加输出层(10分类)
    ])
    
#打印每层参数信息 
model.summary()  

#编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(), #使用adam优化器
    loss = 'categorical_crossentropy', #交叉熵损失函数
    metrics=['accuracy'] #评价函数
)

def lrfn(epoch):
    LR_START = 0.00001 #初始学习率
    LR_MAX = 0.0004 #最大学习率
    LR_MIN = 0.00001 #学习率下限
    LR_RAMPUP_EPOCHS = 5 #上升过程为5个epoch
    LR_SUSTAIN_EPOCHS = 0 #学习率保持不变的epoch数
    LR_EXP_DECAY = .8 #指数衰减因子
    
    if epoch < LR_RAMPUP_EPOCHS: #第0-第5个epoch学习率线性增加
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS: #不维持
        lr = LR_MAX
    else: #第6-第15个epoch学习率指数下降
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr

#绘制学习率曲线
rng = [i for i in range(epochs)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
#print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

#使用tensorflow中的回调函数LearningRateScheduler设置学习率
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

#保存最优模型
checkpoint = ModelCheckpoint(
                                filepath='./save_weights/myefficientnet.ckpt', #保存模型的路径
                                monitor='val_acc',  #需要监视的值
                                save_weights_only=False, #若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
                                save_best_only=True, #当设置为True时，监测值有改进时才会保存当前的模型
                                mode='auto',  #当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min，在auto模式下，评价准则由被监测值的名字自动推断
                                period=1 #CheckPoint之间的间隔的epoch数
                            )
               
#开始训练
history = model.fit(x=train_data_gen,   #输入训练集
                    steps_per_epoch=total_train // batch_size, #一个epoch包含的训练步数
                    epochs=epochs, #训练模型迭代次数
                    validation_data=val_data_gen,  #输入验证集
                    validation_steps=total_val // batch_size, #一个epoch包含的训练步数
                    callbacks=[checkpoint, lr_schedule]) #执行回调函数

#保存训练好的模型权重                    
model.save_weights('./save_weights/myefficientnet.ckpt',save_format='tf')

# 记录训练集和验证集的准确率和损失值
history_dict = history.history
train_loss = history_dict["loss"] #训练集损失值
train_accuracy = history_dict["accuracy"] #训练集准确率
val_loss = history_dict["val_loss"] #验证集损失值
val_accuracy = history_dict["val_accuracy"] #验证集准确率


plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')


plt.figure()
plt.plot(range(epochs), train_accuracy, label='train_accuracy')
plt.plot(range(epochs), val_accuracy, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm)) #计算准确率
    misclass = 1 - accuracy #计算错误率
    if cmap is None:
        cmap = plt.get_cmap('Blues') #颜色设置成蓝色
    plt.figure(figsize=(10, 8)) #设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap) #显示图片
    plt.title(title) #显示标题
    plt.colorbar() #绘制颜色条

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90) #x坐标标签旋转90度
        plt.yticks(tick_marks, target_names) #y坐标

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm,2) #对数字保留两位小数
        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #将cm.shape[0]、cm.shape[1]中的元素组成元组，遍历元组中每一个数字
        if normalize: #标准化
            plt.text(j, i, "{:0.2f}".format(cm[i, j]), #保留两位小数
                     horizontalalignment="center",  #数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  #设置字体颜色
        else:  #非标准化
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",  #数字在方框中间
                     color="white" if cm[i, j] > thresh else "black") #设置字体颜色

    plt.tight_layout() #自动调整子图参数,使之填充整个图像区域
    plt.ylabel('True label') #y方向上的标签
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass)) #x方向上的标签
    plt.show() #显示图片

#读取'Common Name'列的猴子类别，并存入到labels中
cols = ['Label','Latin Name', 'Common Name','Train Images', 'Validation Images']
labels = pd.read_csv("../input/10-monkey-species/monkey_labels.txt", names=cols, skiprows=1)
labels = labels['Common Name']

# 预测验证集数据整体准确率
Y_pred = model.predict_generator(val_data_gen, total_val // batch_size + 1)
# 将预测的结果转化为one hit向量
Y_pred_classes = np.argmax(Y_pred, axis = 1)
# 计算混淆矩阵
confusion_mtx = confusion_matrix(y_true = val_data_gen.classes,y_pred = Y_pred_classes)
# 绘制混淆矩阵
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)

