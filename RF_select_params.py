#import libs 导入机器学习库sklearn，数据处理库numpy、pandas，绘图库matplotlib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.tree import export_graphviz
from sklearn.preprocessing import Imputer
from matplotlib import pyplot as plt
import matplotlib

#set fonts 将绘图的默认设置中加入中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei','Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

#import datasets导入样本数据集
features = pd.read_excel("D:/data/feature_all_miss_m.xlsx")
X1 = features.values
imp = Imputer(missing_values='NaN', strategy='median', axis=0)#对缺失值用中位数处理
imp.fit(X1)
X1 = imp.transform(X1)
label = pd.read_excel("D:/data/label_3_miss.xlsx")
y = np.ravel(label.values)


#split data 将样本数数据集分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(
X1, y, test_size=0.2,random_state =42)


fig=plt.figure(figsize=(6,7),dpi=600)#设置绘图的画布

#接下来五个循环分别对n_estimator，max_depth，max_features, min_samples_leaf, min_samples_split参数调参

#对n_estimator进行调参
es_range= range(10,101,10)                   
score_list_1 = []                        
for n_estimators in es_range:

    #建立模型参数
    clf =RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, max_depth=6,
        max_features=8, min_samples_split=2, random_state=1)
                         
    #训练模型
    clf.fit(X_train,y_train)

    #模型预测
    y_pred = clf.predict(X_test)
    #print(classification_report(y_test, y_pred))显示预测结果报告
    score = clf.score(X_test, y_test)#模型的准确率
    score_list_1.append(score)
    print ("score is(n_estimator): " + str(score))
    #print(clf.feature_importances_)显示特征重要性
    print('===================================================================')
max_score = max(score_list_1)#找到最大准确率

index = score_list_1.index(max_score)#找到预测最高分对应的位置
best_estimator = es_range[index]#找到最高分对应的参数值

#visualization 绘图，横坐标为参数的取值范围，纵轴为准确率
ax1=fig.add_subplot(321)
ax1.plot(es_range,score_list_1,color = 'black',marker='o',markersize=4)
ax1.set_xlabel('n_estimators',fontsize=12,fontproperties='Times New Roman')
ax1.set_ylabel(u'准确率',fontsize=12,fontproperties='SimHei')
#ax1.text(es_range[index],max_score, (es_range[index],max_score),
         # ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
print()
print()


#对max_depth进行调参
depth_range= range(2,13,1)                   
score_list_2= []                        
for max_depth in depth_range:
    #建立模型参数，并应用上一循环找到的最优参数值
    clf =RandomForestClassifier(n_estimators=best_estimator, min_samples_leaf=2, max_depth=max_depth,
        max_features=6, min_samples_split=2, random_state=1)
                         
    #训练模型
    clf.fit(X_train,y_train)

    #模型预测
    y_pred = clf.predict(X_test)
    #print(classification_report(y_test, y_pred))
    score = clf.score(X_test, y_test)#模型的准确率
    ##sum_score += score
    score_list_2.append(score)
    print ("score is(max_depth): " + str(score))
    #print(clf.feature_importances_)
    print('===================================================================')
max_score = max(score_list_2)#找到最大准确率

index = score_list_2.index(max_score)#找到预测最高分对应的位置
best_depth = depth_range[index]#找到最高分对应的参数值

#绘图，横坐标为参数的取值范围，纵轴为准确率
ax2=fig.add_subplot(322)
ax2.plot(depth_range,score_list_2,color = 'black',marker='o',markersize=4)
ax2.set_xlabel('max_depth',fontsize=12,fontproperties='Times New Roman')
ax2.set_ylabel(u'准确率',fontsize=12,fontproperties='SimHei')
#ax2.text(depth_range[index],max_score, (depth_range[index],max_score),
         # ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
print()
print()

#对max_features进行调参
features_range= range(1,28,1)                   
score_list_3= []                        
for max_features in features_range:
    #建立模型参数，并应用上一循环找到的最优参数值
    clf =RandomForestClassifier(n_estimators=best_estimator, min_samples_leaf=2, max_depth=best_depth,
        max_features=max_features, min_samples_split=2, random_state=1)
                         
    #训练模型
    clf.fit(X_train,y_train)

    #模型预测
    y_pred = clf.predict(X_test)
    #print(classification_report(y_test, y_pred))
    score = clf.score(X_test, y_test)#模型的准确率
    ##sum_score += score
    score_list_3.append(score)
    print ("score is(max_features): " + str(score))
    #print(clf.feature_importances_)
    print('===================================================================')
	
max_score = max(score_list_3)#找到最大准确率
index = score_list_3.index(max_score)#找到预测最高分对应的位置
best_features = features_range[index]#找到最高分对应的参数值

#绘图，横坐标为参数的取值范围，纵轴为准确率
ax3=fig.add_subplot(323)
ax3.plot(features_range,score_list_3,color = 'black',marker='o',markersize=4)
ax3.set_xlabel('max_features',fontsize=12,fontproperties='Times New Roman')
ax3.set_ylabel(u'准确率',fontsize=12,fontproperties='SimHei')
#ax3.text(features_range[index],max_score, (features_range[index],max_score),
          #ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
print()
print()

#对min_samples_leaf进行调参
min_samples_leaf_range= range(1,22,1)                   
score_list_4= []                        
for min_samples_leaf in min_samples_leaf_range:
    #建立模型参数，并应用上一循环找到的最优参数值
    clf =RandomForestClassifier(n_estimators=best_estimator, min_samples_leaf=min_samples_leaf, max_depth=best_depth,
        max_features=best_features, min_samples_split=2, random_state=1)
                         
    #训练模型
    clf.fit(X_train,y_train)

    #模型预测
    y_pred = clf.predict(X_test)
    #print(classification_report(y_test, y_pred))
    score = clf.score(X_test, y_test)#模型的准确率
    ##sum_score += score
    score_list_4.append(score)
    print ("score is(min_samples_leaf): " + str(score))
    #print(clf.feature_importances_)
    print('===================================================================')
	
max_score = max(score_list_4)#找到最大准确率
index = score_list_4.index(max_score)#找到预测最高分对应的位置
best_min_samples_leaf = min_samples_leaf_range[index]#找到最高分对应的参数值

#绘图，横坐标为参数的取值范围，纵轴为准确率
ax4=fig.add_subplot(324)
ax4.plot(min_samples_leaf_range,score_list_4,color = 'black',marker='o',markersize=4)
ax4.set_xlabel('min_samples_leaf',fontsize=12,fontproperties='Times New Roman')
ax4.set_ylabel(u'准确率',fontsize=12,fontproperties='SimHei')
#ax4.text(min_samples_leaf_range[index],max_score, (min_samples_leaf_range[index],max_score),
          #ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
print()
print()

#对min_samples_split进行调参
min_samples_split_range= range(2,22,1)                   
score_list_5= []                        
for min_samples_split in min_samples_split_range:
    #建立模型参数，并应用上一循环找到的最优参数值
    clf =RandomForestClassifier(n_estimators=best_estimator, min_samples_leaf=best_min_samples_leaf, max_depth=best_depth,
        max_features=best_features, min_samples_split=min_samples_split, random_state=1)
                         
    #训练模型
    clf.fit(X_train,y_train)

    #模型预测
    y_pred = clf.predict(X_test)
    #print(classification_report(y_test, y_pred))
    score = clf.score(X_test, y_test)#模型的准确率
    ##sum_score += score
    score_list_5.append(score)
    print ("score is(min_samples_split): " + str(score))
    #print(clf.feature_importances_)
    print('===================================================================')

max_score = max(score_list_5)#找到最大准确率
index = score_list_5.index(max_score)#找到预测最高分对应的位置
best_min_samples_split = min_samples_split_range[index]#找到最高分对应的参数值

#绘图，横坐标为参数的取值范围，纵轴为准确率
ax5=fig.add_subplot(325)
ax5.plot(min_samples_split_range,score_list_5,color = 'black',marker='o',markersize=4)
ax5.set_xlabel('min_samples_split',fontsize=12,fontproperties='Times New Roman')
ax5.set_ylabel(u'准确率',fontsize=12,fontproperties='SimHei')
#ax5.text(min_samples_split_range[index],max_score, (min_samples_split_range[index],max_score),
         # ha='left', va='baseline', fontsize=13,fontname='Times New Roman')
print()
print()

#visualization 将随机森林中第二颗决策树可视化导出
export_graphviz(clf.estimators_[1],feature_names=features.columns,filled=False,
                    rounded=True,precision=2,proportion=True,out_file='tree.dot')

plt.tight_layout(2)#设置图片排列格式
plt.savefig('随机森林调参2.jpg')#保存图片
