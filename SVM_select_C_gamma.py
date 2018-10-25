#import libs 导入Sklearn库建立机器学习模型，numpy、pandas库处理数据，matplotlib库画图
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#用中文输出需要设置的参数
##plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
##plt.rcParams['axes.unicode_minus']=False

#import datasets 导入[油气数据]综合样本数据集，并进行数据预处理
features = pd.read_excel('D:/data/feature_all_miss_m.xlsx')
X = features.values
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X1 = imp.transform(X)
min_max_scaler = preprocessing.MinMaxScaler()
X2 = min_max_scaler.fit_transform(X1)

label = pd.read_excel('D:/data/label_3_miss.xlsx')
y = np.ravel(label.values)

score1=[]#C一定，存储不同gamma对应的准确率
score2=[]#gamma一定，存储不同C对应的准确率

##split data 将样本数据集分为80%训练集，20%测试集
X_train, X_test, y_train, y_test = train_test_split(
X2, y, test_size=0.2, random_state= 81)


Gamma=[0.0001, 0.001, 0.005, 0.008, 0.01, 0.05, 0.1,   1]
C =   [     1,    10,   100,   200,  300,  1e3 ,1e4, 1e5]

#给定C，研究gamma对准确率的影响
for g in Gamma:
    #train
    clf = SVC(kernel='rbf',C=100,gamma=g)
    clf = clf.fit(X_train, y_train)


    #predict
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(clf.score(X_test, y_test))
    print('.........................................')
    score1.append(clf.score(X_test, y_test))
print('--------------------------------------------------------------')

#给定gamma，研究C对准确率的影响
for c in C:
    clf2 = SVC(kernel='rbf',C=c,gamma=0.05)
    clf2 = clf2.fit(X_train, y_train)
    

    #predict
    y_pred = clf2.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(clf2.score(X_test, y_test))
    print('.........................................')
    score2.append(clf2.score(X_test, y_test))
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')   
print("score1: " + str(score1))
print("max score1 is: "+ str(max(score1)))
print('==========================')
print("score2: " + str(score2))
print("max score2 is: "+ str(max(score2)))

fig, axes = plt.subplots(1,2,figsize=(10,5),dpi=600)

axes[0].semilogx(Gamma,score1,marker='o',markerfacecolor='black',color='black')
axes[0].set_xlabel('gamma',fontsize=15,fontname='Times New Roman')

axes[0].text(0.05, 0.9, (0.05,0.9),ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
axes[0].text(0.0001, 0.89, ('C=100'),ha='left', va='baseline', fontsize=14,fontname='Times New Roman')

axes[0].set_ylabel(u'准确率',fontsize=17,fontproperties='SimHei')



axes[1].semilogx(C,score2,marker='o',markerfacecolor='black',color='black')
axes[1].set_xlabel('C',fontsize=15,fontname='Times New Roman')

axes[1].text(1, 0.905, ('gamma=0.05'),ha='left', va='baseline', fontsize=14,fontname='Times New Roman')


axes[1].text(200, 0.917, (200,0.917),ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
axes[1].set_ylabel(u'准确率',fontsize=17,fontproperties='SimHei')    
plt.tight_layout(4)
plt.savefig("example.jpg")

