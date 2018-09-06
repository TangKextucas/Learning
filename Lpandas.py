import pandas as pd
import numpy as np
# # df = pd.DataFrame(np.arange(20).reshape(4, 5), columns=list('abcde'))
# # print(df)
# # df['f'] = df['a'] + df['b']     # 两列相加，结果存在新的一列
# # print(df)
# # df['f'] = np.sqrt(df['a']) + np.sqrt(df['b'])
# # print(df)
# # print(df.applymap(lambda x: x + 10))
# # print(df['f'].map(lambda x: '%.2f' % x))
# # df['g'] = df['a'].apply(lambda x: x-10)
# # print(df)
# # df1 = pd.DataFrame(np.arange(20).reshape(4, 5), columns=list('abcde'), index=list('aaac'))
# # print(df1)
# # print(df1.groupby(['a']).count())
# # df = pd.DataFrame({'key1': list('abdcbcadcabdcbacdbac'),
# #                    'key2': list('xzyxyzyzxyzyxzyxyzyx'),
# #                    'data1': np.random.randn(20),
# #                    'data2': np.random.randn(20)})
# #
# # print(df)
# # a = pd.DataFrame(df['key1'])
# # a['gap'] = (df['data2'] - df['data1'])*10
# # print(a)
# # df.drop(['data1', 'data2'], axis=1, inplace=True)
# # a = pd.DataFrame(df['key1'])
# # print(a)
# # print(df[df['data1'] > 0])
# # print(df[df['data1'] > 0].groupby(['key1']).count())
# # a = pd.DataFrame({'device': list('aaaaabbbbccc'),
# #                   'app': list('xxxyyxxyyxxy'),
# #                   'start': [1, 5, 13, 2, 6, 3, 7, 5, 10, 3, 9, 7],
# #                   'close': [2, 9, 24, 3, 8, 4, 9, 6, 13, 6, 10, 9]})
# # # print(a)
# # a['gap'] = a['close'] - a['start']
# # print(a)
# # i = dict(list(a.groupby(['device'])))
# # # print(x)
# # # print(type(x['x']))
# # device_list = []
# # app_list = []
# # gap_list = []
# # for k1 in i:
# #     j = dict(list(i[k1].groupby(['app'])))
# #     for k2 in j:
# #         gap_list.append(np.sum(j[k2]['gap']))
# #         app_list.append(k2)
# #         device_list.append(k1)
# # timeall = pd.DataFrame({'device': device_list, 'app': app_list, 'totaltime': gap_list})
# # print(timeall)
# # tmp = pd.DataFrame(x)
# # print(tmp)
# # dict = {'a': 1, 'b': 2, 'c': 3}
# # for k, v in dict.items():
# #     timeall[k] = v
# # print(timeall)
#
# #
# import time
# import datetime
# a = datetime.datetime.now()
# print(a)
# print(time.mktime(a.timetuple()))
# print(datetime.datetime.fromtimestamp(1490944691))
# a = pd.DataFrame({'device': list('123'), 'app': ['a,b,c', 'b,c,d,f', 'a,e,f']})
# # print(a)
# b = pd.DataFrame({'app': list('abcdef'), 'brand': ['dust2', 'mirage', 'overpass', 'inferno', 'nuke', 'cache']})
# # print(b)
# dict1 = {}
# for row in b.itertuples(index=True, name='Pandas'):
#     key = getattr(row, 'app')
#     value = getattr(row, 'brand')
#     dict1[key] = value
# # print(dict1)
# def f(applist):
#     ans = []
#     for i in applist.split(','):
#         ans.append(dict1[i])
#     return ans
# # print(f('a,c,d,f'))
# a['brand'] = a['app'].apply(lambda x: f(x))
# a.drop(['app'], axis=1, inplace=True)
# print(a)
# a = pd.Series(list(map(int, '1209384758264081373298740123478374657483920178347678902137')))
# print(a)
# print(a.skew())
# print(a.kurt())
# print(a.describe(percentiles=[.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]))
# a = pd.DataFrame({'a': list('ewfewf'), 'b': list('iojijo')})
# b = pd.DataFrame({'c': list('123456'), 'd': list('4670jo')})
# print(pd.concat([a, b], axis=1))
# a = pd.DataFrame({'device': list('aaaabbb'), 'app': list('1234159'), 'time': [23, 45, 67, 54, 24, 88, 33]})
# print(a)
# a.sort_values(by=['time'], ascending=False, inplace=True)
# print(a)
# print(list(zip(a.loc[2, ])))
# tmp = dict(list(a.groupby(['device'], axis=1)))
# ans = pd.DataFrame()
# for v in tmp.values():
# print(a)
# dict1 = {}
# for row in a.itertuples(index=True, name='Pandas'):
#     key = getattr(row, 'time')
#     value = getattr(row, 'app')
#     dict1[key] = value
# a.drop(['app'], axis=1, inplace=True)
# fuck = (a.groupby(['device']).max())['time']
# # device
# # a    67
# # b    88
# fuck = fuck.apply(lambda x: dict1[x])
# print(fuck)
# ans = pd.DataFrame(a['device'])

# print(ans)
# a = np.array([1, 2, 3])
# d = {1: 'a', 2: 'b', 3: 'c'}
# b = d[a]
# print(b)

# a = {}
# for i in range(24):
#     a[i] = np.random.randint(10, 100)
# print(a)
# v = list(a.values())
# print(np.mean(v))
# # print(1530039.1666666667/72728)
# print(810464.157756769/72728)       #11.143770731448258


# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
#
# def getmu(dis):
#     a = list(dis.values())
#     N = np.sum(a)
#     ans = np.sum(i*a[i] for i in range(24))
#     return ans/N
# def getsigma(dis):
#     a = list(dis.values())
#     onestd = np.std(a)
#     return onestd
# mu = getmu(a)
# sigma = getsigma(a)
# num_bins = 24  # 直方图柱子的数量
# x = np.array(a.keys())
# # y = np.array(a.values())
# n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
# # 直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
# y = mlab.normpdf(bins, mu, sigma)  # 拟合一条最佳正态分布曲线y
# plt.plot(bins, y, 'r--')  # 绘制y的曲线
# plt.xlabel('sepal-length')  # 绘制x轴
# plt.ylabel('Probability')  # 绘制y轴
# plt.title(r'Histogram : $\mu=5.8433$,$\sigma=0.8253$')  # 中文标题 u'xxx'
# plt.subplots_adjust(left=0.15)  # 左边距
# plt.show()
# a = pd.DataFrame({'nice': list('jsfbh')}, index=[4, 7, 3, 6, 1])
# 意思就是，做的特征的表的device必须完全包含训练集里面的device,才不会出现缺失值
# 如果没有完全包含，空位就出现缺失值
# b = pd.DataFrame({'cache': list('acdebqwe')}, index=[1, 3, 4, 7, 0, 9, 6, 8])
# d = pd.DataFrame({'dust': list('lkdfjklgjl')}, index=[0, 6, 8, 3, 5, 4, 10, 7, 2, 4])
# c = a.join([b, d])
# 前后顺序没有关系，只是改变列的排列顺序
# print(a)
# print(b)
# print(d)
# print(c)
#   nice
# 4    j
# 7    s
# 3    f
# 6    b
# 1    h
#   cache
# 1     a
# 3     c
# 4     d
# 7     e
# 0     b
# 9     q
# 6     w
# 8     e
#    dust
# 0     l
# 6     k
# 8     d
# 3     f
# 5     j
# 4     k
# 10    l
# 7     g
# 2     j
# 4     l
#   nice cache dust
# 1    h     a  NaN
# 3    f     c    f
# 4    j     d    k
# 4    j     d    l
# 6    b     w    k
# 7    s     e    g
# a = pd.DataFrame({'ind': list('wdasdawdsdawdasd'), 'a': list(map(int, '9183746394267501')), 'b': list(map(int, '4893562176843519'))}, index=range(21, 37))
# print(type(a))
# a = a.sort_values(['a'], ascending=False).reset_index(drop=True)
# print(a)
# print(a.loc[1, ])
# print(len(a.index))
# tmp = dict(list(a.groupby(['ind'])))
# print(tmp)
# print(a.groupby(['ind']).min()['a'])
# print(a.groupby(['ind']).max()['b'] - a.groupby(['ind']).min()['a'])
# from sklearn.preprocessing import LabelEncoder
# a = [['dui', 'nan', 'nan', 'gefff'], ['dsffg', 'asdsad', 'effe']]
# le = LabelEncoder()
# a = le.fit_transform(a)
# print(a)            # [1 0]
# data = pd.DataFrame({'a': [1, 2, 3, np.nan, 2], 'b': [np.nan, 2, 4, 4, 8]})
# print(data)
# data.dropna(inplace=True, subset=['a'])
# print(data)
# data[data['a'].isnull()]['a'] = data['a'].mode()
# modea = data['a'].mode()[0]
# data['a'] = data['a'].replace(np.nan, modea)
"""好不容易发现的坑:缺失值按列填充众数的方法"""
# data.fillna({'a': data['a'].mode()[0], 'b': data['b'].mode()[0]}, inplace=True)
# print(data)
# a = int(data.groupby(['app']).sum().loc['nice'])
data = pd.DataFrame({'a': [1, 3, 6, 2, 9], 'b': ['w', 'a', 's', 'd', 'p']})
print(data)
print('w' not in data['b'].values)
print(2**32)

