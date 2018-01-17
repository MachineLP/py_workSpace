# -*- coding: utf-8 -*-
"""
Created on 2017 11.17
@author: liupeng
"""

import pandas as pd  
import numpy as np  
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler  


from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc,  precision_recall_curve
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# ROC曲线。
def c_roc(y_pred, y_test):
    mean_tpr = 0.0  
    mean_fpr = np.linspace(0, 1, 1000)  
    all_tpr = [] 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred )
    mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
    mean_tpr[0] = 0.0                               #初始处为0  
    roc_auc = auc(fpr, tpr)  
    plt.plot(fpr, tpr, lw=1, label='LogisticRegression %d (area = %0.2f)' % (0, roc_auc)) 
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating characteristic example')  
    plt.legend(loc="lower right")  
    plt.show() 
# 箱线图。
def c_boxplot(data):

    # Creating boxplots 
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # 合并 X, y。
    #data = np.hstack((X,np.reshape(y,[-1, 1])))
    bp = ax.boxplot(data, patch_artist=True)
    #plt.boxplot(X, sym="o", whis=1.5)
    
    for box in bp['boxes']:
        # change outline color
        box.set( color='#7570b3', linewidth=2)
        # change fill color
        box.set( facecolor = '#1b9e77' )
    
    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='#7570b3', linewidth=2)
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='#b2df8a', linewidth=2)
    
    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
    
    # Save the figure
    # fig.savefig('fig1.png', bbox_inches='tight')
    plt.show()


###获取数据和训练---------------------------------------------------------###
# 从 csv中读取训练数据和标签。
train_data = pd.read_csv("train_data.csv")  
print (train_data)

# 分离 训练样本 和 标签。
X = train_data.values[0::, 0:16]
y = train_data.values[0::, 16].astype(np.int)

# 仅取 5000 个样本。
X = X[40000:45000]
y = y[40000:45000]

print (X)
print (y)


# 5折交叉验证。
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.5, random_state = 0)
print (len(X_train), len(X_test), len(y_train), len(y_test))
#使用6折交叉验证，并且画ROC曲线，也可以用下面产生5折交叉验证，产生5个集合。
#cv = StratifiedKFold(y, n_folds=5) 

################LogisticRegression  
from sklearn.linear_model import LogisticRegression   
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# 预测
y_pred = classifier.predict(X_test)
print ("========>", y_pred)
y_train_pred = classifier.predict(X_train)
# 计算得分
scores = classifier.score(X_test , y_test)
print ('scores:', scores)

# 画ROC曲线
c_roc(y_pred, y_test)

a = [0,0,0]
b = [0,0,0]
p,r,t = precision_recall_curve(a, b)
print (">>>>>>", p,r,t)
# 合并 X, y。
data = np.hstack((X,np.reshape(y,[-1, 1])))
# 画 boxplot
c_boxplot(data)

# 计算混淆矩阵
confusion_matrix=confusion_matrix(y_test,y_pred)
print (confusion_matrix)
# 显示混淆矩阵
plt.matshow(confusion_matrix)
plt.title(u'混淆矩阵')
plt.colorbar()
plt.ylabel(u'实际类型')
plt.xlabel(u'预测类型')
plt.show()


###-------------------------------------------------------------------------------------###
# 以上的交叉验证只需要下面即可。
from sklearn.cross_validation import cross_val_score
# 在X，y原始数据上，计算5折交叉验证的准确率
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
scores = classifier.score(X_test , y_test)
scores1 = np.mean(scores)
print ('LogisticRegression Accuracy：',np.mean(scores), scores)
y_pred = classifier.predict(X_test)
# 合并 X, y。
data = np.hstack((X_test,np.reshape(y_test,[-1, 1]), np.reshape(y_pred,[-1, 1])))
# 画 boxplot
c_boxplot(data)


################AdaBoostClassifier 
from sklearn.ensemble import AdaBoostClassifier  

classifier = AdaBoostClassifier(n_estimators=100) #迭代100次  

classifier.fit(X_train, y_train)
scores = classifier.score(X_test , y_test)
scores2 = np.mean(scores)
print ('AdaBoostClassifier Accuracy ：',np.mean(scores), scores)
y_pred = classifier.predict(X_test)
# 合并 X, y。
data = np.hstack((X_test,np.reshape(y_test,[-1, 1]), np.reshape(y_pred,[-1, 1])))
# 画 boxplot
c_boxplot(data)


################RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100) #迭代100次  

classifier.fit(X_train, y_train)
scores = classifier.score(X_test , y_test)
scores3 = np.mean(scores)
print ('RandomForestClassifier Accuracy ：',np.mean(scores), scores)   
y_pred = classifier.predict(X_test)
# 合并 X, y。
data = np.hstack((X_test,np.reshape(y_test,[-1, 1]), np.reshape(y_pred,[-1, 1])))
# 画 boxplot
c_boxplot(data)


###############radial basis functions, and support vector machines
from sklearn import svm

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, y_train)
scores = classifier.score(X_test , y_test)
scores4 = np.mean(scores)
print ('support vector machines(linear) Accuracy ：',np.mean(scores), scores)   
y_pred = classifier.predict(X_test)
# 合并 X, y。
data = np.hstack((X_test,np.reshape(y_test,[-1, 1]), np.reshape(y_pred,[-1, 1])))
# 画 boxplot
c_boxplot(data)


classifier = svm.SVC(kernel='poly', degree=3)

classifier.fit(X_train, y_train)
scores = classifier.score(X_test , y_test)
scores5 = np.mean(scores)
print ('support vector machines(poly) Accuracy ：',np.mean(scores), scores)   
y_pred = classifier.predict(X_test)
# 合并 X, y。
data = np.hstack((X_test,np.reshape(y_test,[-1, 1]), np.reshape(y_pred,[-1, 1])))
# 画 boxplot
c_boxplot(data)


classifier = svm.SVC(kernel='rbf')

classifier.fit(X_train, y_train)
scores = classifier.score(X_test , y_test)
scores6 = np.mean(scores)
print ('support vector machines(rbf) Accuracy ：',np.mean(scores), scores)   
y_pred = classifier.predict(X_test)
# 合并 X, y。
data = np.hstack((X_test,np.reshape(y_test,[-1, 1]), np.reshape(y_pred,[-1, 1])))
# 画 boxplot
c_boxplot(data)


# 各准确率的比较
data = np.hstack((np.reshape(scores1,[-1, 1]), np.reshape(scores2,[-1, 1]),np.reshape(scores3,[-1, 1]),np.reshape(scores4,[-1, 1]),np.reshape(scores5,[-1, 1]), np.reshape(scores5,[-1, 1])))
# 画 boxplot
c_boxplot(data)


































