import numpy as np
# 预处理：决定了算法准确率的上界
from sklearn import preprocessing
X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
stdscaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
# print(stdscaler.transform(X_train))
# 标准化：通过平移中心,除以方差。得到平均值为0,方差为1的标准正态分布
# [[ 0.         -1.22474487  1.33630621]
#  [ 1.22474487  0.         -0.26726124]
#  [-1.22474487  1.22474487 -1.06904497]]
min_max_scaler = preprocessing.MinMaxScaler([0, 4])
# print(min_max_scaler.fit_transform(X_train))
# 数据缩放到某个范围
# [[2.         0.         4.        ]
#  [4.         2.         1.33333333]
#  [0.         4.         0.        ]]
binarizer = preprocessing.Binarizer().fit(X_train)
# print(binarizer.transform(X_train))
# 二值化：将连续的数值特征转换为01两种类型的数据，主要参数:阈值threshold
# [[1. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]]
# 类别编码主要是用来处理文本数据
# one-hot:每个类别变为一个特征，把一列变为n列，n等于这一列的类别数
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
# print(le.transform(["tokyo", "tokyo", "paris"]))
# 每个类别用一个数字表示,一个特征处理后的数据仍是一列数据
# [2 2 1]
lb = preprocessing.LabelBinarizer()
lb.fit(["paris", "paris", "tokyo", "amsterdam"])
# print(lb.transform(["tokyo", "tokyo", "paris"]))
# 这两个函数也能处理int型数据，但是都一次只能处理一列数据，对整个数据集需要循环
# [[0 0 1]
#  [0 0 1]
#  [0 1 0]]
# 缺失值处理
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit([[1, 2],
         [np.nan, 3],
         [7, 6]])
X = [[np.nan, 2],
     [6, np.nan],
     [7, 6]]
# print(imp.transform(X))
# 字符类型缺失值一般用pandas来解决
# [[4.         2.        ]
#  [6.         3.66666667]
#  [7.         6.        ]]
# 导入sklearn自带的数据集
from sklearn import datasets
X = datasets.load_digits()['data']
Y = datasets.load_digits()['target']
# print(X.shape, Y.shape)
# (1797, 64) (1797,)
# 单变量因素分析:通过一定的指标判断单个特征与目标值的关联关系
# 回归指标：f_regression, mutual_info_regression
# 分类指标：卡方分布chi2, f_classif, mutual_info_classif
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_classif
# select = SelectKBest(mutual_info_classif, 40)
# X_new = select.fit_transform(X, Y)
# print(select.scores_)
# [0.         0.13322524 0.37420598 0.19880696 0.13487987 0.26621657
#  0.19001271 0.03923858 0.0127144  0.26908678 0.35090464 0.14495486
#  0.21082142 0.34115807 0.1260321  0.01506455 0.01605465 0.16379622
#  0.29649237 0.26372015 0.35501404 0.42540483 0.19528575 0.03673796
#  0.02408227 0.26464686 0.41284368 0.28265491 0.39285778 0.2990641
#  0.38707086 0.         0.         0.43326716 0.43415639 0.27758372
#  0.35777084 0.27612534 0.3714942  0.         0.01255832 0.27580768
#  0.41323823 0.38030483 0.28627817 0.17656711 0.33448573 0.03024138
#  0.00783296 0.05189728 0.26019857 0.23918916 0.19791448 0.32698077
#  0.3783966  0.06551915 0.00455891 0.105576   0.37179672 0.19794735
#  0.25850554 0.34974424 0.26267263 0.0776573 ]
# print(X_new.shape)
# (1797, 40)  最后选出了40个特征
from sklearn import feature_selection
from sklearn import linear_model
# 递归特征消除：利用一个模型来评价每个特征的权重，删除不重要的一个，然后继续迭代
select = feature_selection.RFE(estimator=linear_model.LinearRegression(), n_features_to_select=40)
# print(select.fit_transform(X, Y))
# [[ 0.  9.  0. ... 13.  0.  0.]
#  [ 0. 13.  0. ... 11. 10.  0.]
#  [ 0. 15.  0. ...  3. 16.  0.]
#  ...
#  [ 0. 15.  0. ...  9.  6.  0.]
#  [ 0.  7.  0. ... 12. 12.  0.]
#  [ 0.  8.  0. ... 12. 12.  0.]]
# print(select.ranking_)
# 最终选择的表示为1
# [25  1 16  5  1 21  4 20  1 10  1  1  1  1  1  1  1 11  1  1  1  1  1  1
#   1  1  6  1  1  1  8  1 23  1 12  1  2  3 13 24  1  1  7 18  1  1 14  1
#   1  9 19  1  1  1  1  1  1  1 17  1 15  1 22  1]
# 利用模型直接消除：利用一个模型来评价每个特征的重要性，删除重要性低于阈值的特征
# 直接使用模型，默认是阈值均值
select = feature_selection.SelectFromModel(linear_model.LogisticRegression(penalty="l1", C=0.1))
X_new = select.fit_transform(X, Y)
# print(X_new.shape)
# (1797, 48)
# 将数据分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
# K近邻法(监督学习)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
# 使用交叉验证来调参，有0.3比例的简单交叉验证，也有K折交叉验证
from sklearn.model_selection import GridSearchCV
par = {'algorithm': ['auto', 'brute'], 'n_neighbors': np.arange(5, 10)}
grid = GridSearchCV(KNeighborsClassifier(), par, cv=3)
# grid = GridSearchCV(cv=3, error_score='raise',
#                     estimator=KNeighborsClassifier(
#                         algorithm='auto', leaf_size=30, metric='minkowski',
#                         metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform'),
#                     fit_params=None, iid=True, n_jobs=1,
#                     param_grid={'algorithm': ['auto', 'brute'], 'n_neighbors': np.array([5, 6, 7, 8, 9])},
#                     pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#                     scoring=None, verbose=0)
grid.fit(X_train, Y_train)
# print(grid.cv_results_['mean_test_score'])
# 得到测试分数
# [0.98254364 0.98004988 0.98004988 0.97339983 0.97506234 0.98254364
#  0.98088113 0.98004988 0.97339983 0.97506234]
model = grid.best_estimator_
# print(model)
# 得到表现最好的参数对应的模型
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
# 使用k折交叉验证(每个都是先训练再验证)来评价模型的效果
from sklearn.model_selection import cross_validate
result = cross_validate(model, X_train, Y_train)
# print(result)
# {'fit_time': array([0., 0., 0.]),
#  'score_time': array([0.04691172, 0.04691291, 0.04687643]),
#  'test_score': array([0.97530864, 0.975     , 0.98241206]),
#  'train_score': array([0.9887218 , 0.99252802, 0.98136646])}
# 对结果进行进一步的指标衡量(分类衡量)
from sklearn import metrics
# print(metrics.accuracy_score(Y_test, model.predict(X_test)))
# 0.9882154882154882
# print(metrics.classification_report(Y_test, model.predict(X_test)))
#              precision    recall  f1-score   support
#
#           0       1.00      0.99      0.99        68
#           1       0.94      1.00      0.97        58
#           2       1.00      1.00      1.00        49
#           3       1.00      0.98      0.99        63
#           4       0.98      0.94      0.96        68
#           5       1.00      0.98      0.99        57
#           6       1.00      0.98      0.99        62
#           7       0.95      1.00      0.98        60
#           8       0.94      0.96      0.95        53
#           9       0.98      0.96      0.97        56
#
# avg / total       0.98      0.98      0.98       594
# SVM、朴素贝叶斯、逻辑回归(监督学习)
# 利用身高、体重、性别预测胖瘦
train_data = [[160, 60, 1], [155, 80, 1], [178, 53, 2], [158, 53, 2], [166, 45, 2],
              [170, 50, 2], [156, 56, 2], [166, 50, 1], [175, 55, 1], [188, 68, 1],
              [159, 41, 1], [166, 70, 1], [175, 85, 1], [188, 98, 1], [159, 61, 2]]
train_target = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1]
test_data = [[166, 45, 2], [172, 52, 1], [156, 60, 1], [150, 70, 2]]
test_target = [0, 0, 1, 1]
from sklearn import svm
clf = svm.SVC()
clf.fit(train_data, train_target)
res = clf.predict(test_data)
# print(res.tolist())
# [0, 1, 1, 1]
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(train_data, train_target)
res = gnb.predict(test_data)
# print(res)
# [0 0 1 1]
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(train_data, train_target)
res = clf.predict(test_data)
# print(res)
# [0 0 1 1]
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(train_data, train_target)
res = clf.predict_proba(test_data)
# print(res)
# [[0.95098932 0.04901068]
#  [0.85844258 0.14155742]
#  [0.18885402 0.81114598]
#  [0.01246901 0.98753099]]
# 聚类(非监督学习)
from sklearn.cluster import KMeans
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
# 输入需要分类的数据和分类数量
KMeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print(KMeans.labels_)
# [0 0 0 1 1 1]
# print(KMeans.cluster_centers_)
# [[1. 2.]
#  [4. 2.]]
# 使用决策树做二分类
from sklearn import tree
X = [[0, 0], [1, 1], [2, 1], [1, 2]]
Y = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
# print(clf.predict([[1.5, 1.]]))
# [0]
# print(clf.predict_proba([[1.5, 1.]]))
# [[1. 0.]]
# 线性判别分析：LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
# print(clf.fit(X, y))
# LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
#               solver='svd', store_covariance=False, tol=0.0001)
# print(clf.predict([[-0.8, -1]]))
# [1]
# 主成分分析：PCA
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
# print(pca.fit(X))
# PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
#   svd_solver='auto', tol=0.0, whiten=False)
# print(pca.explained_variance_ratio_)
# [0.99244289 0.00755711]
# print(pca.singular_values_)
# [6.30061232 0.54980396]
# print(pca.fit_transform(X))
# [[ 1.38340578  0.2935787 ]
#  [ 2.22189802 -0.25133484]
#  [ 3.6053038   0.04224385]
#  [-1.38340578 -0.2935787 ]
#  [-2.22189802  0.25133484]
#  [-3.6053038  -0.04224385]]

# from sklearn.ensemble import forest
# clf = forest.RandomForestClassifier()
# clf.fit
