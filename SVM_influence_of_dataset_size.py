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

#import datasets 导入【油气数据】综合样本数据集，并进行数据预处理
features = pd.read_excel("D:/data/feature_all_miss_m.xlsx")
X1 = features.values
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X1)
X1 = imp.transform(X1)
min_max_scaler = preprocessing.MinMaxScaler()
X2 = pd.DataFrame(min_max_scaler.fit_transform(X1),columns=None)

label = pd.read_excel("D:/data/label_3_miss.xlsx")
y = np.ravel(label.values)


score_list=[]
leny=[]#截取不同的特征数据量
lenx=[]#截取不同的目标数据量

#
for i in range(21):
    #从100个数据量开始，以10为步长递增到300
    line=(i)*10+100
    
    X_train, X_test, y_train, y_test = train_test_split(
    X2.iloc[0:line,:], y[0:line], test_size=0.20, random_state=0)
    leny.append(len(y[0:line]))
    lenx.append(len(X2.iloc[0:line,:]))
	
    #train 训练模型
    param_grid = {'C':[1,    10,   100, 200,  1e3 ,1e4],
              'gamma': [ 0.001, 0.05,  0.01,  0.1,   1,10] }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid,cv=5)
    clf = clf.fit(X_train, y_train)
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    #predict 预测
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    score = clf.score(X_test, y_test)
    score_list.append(score)
    
    print ("score is: " + str(score))
    print('===================================================================')

#print("Average score is: " + str(sum_score/10))
#print()
print (score_list)
print(lenx)
print(leny)
#print()
#print("max score is: "+ str(max(max_list)))

#visualization 作样本数量与准确率关系图
fig, axes = plt.subplots(figsize=(8,6),dpi=600)
axes.scatter(lenx,score_list,marker='o',color='black')
z1=np.polyfit(lenx,score_list,3)
p1 =np.poly1d(z1)
yvals=p1(lenx)
axes.plot(lenx,yvals,color = 'black')
axes.set_xlabel(u'样本数量',fontsize=17,fontproperties='SimHei')
axes.set_ylabel(u'准确率',fontsize=17,fontproperties='SimHei')
print(p1)

plt.show()
#plt.savefig("样本影响.jpg")
