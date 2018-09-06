"""
import datetime
import time


#获取该日期是当年的第几天([0])，是星期几([1])
def getdayinfo(date_str):
    if ':' in date_str:
        t = time.strptime(date_str, "%Y/%m/%d %H:%M")
    else:
        t = time.strptime(date_str, "%Y/%m/%d")
    y, m, d = t[0:3]
    date = datetime.date(y, m, d)
    weekday = date.weekday()        #星期一到星期天分别对应0,1,2,3,4,5,6
    absoluteday = int(date.strftime('%j'))
    return [absoluteday, weekday]
# print(getdayinfo('2018/7/30 22:21'))      #[211, 0]

#获取两个时刻的差，单位是分钟(int)
def getordergap(orderdate_str, confirmdate_str):
    t1 = time.strptime(orderdate_str, "%Y/%m/%d %H:%M")
    y, m, d, h, mi = t1[0:5]
    orderdate = datetime.datetime(y, m, d, h, mi)
    t2 = time.strptime(confirmdate_str, "%Y/%m/%d %H:%M")
    y, m, d, h, mi = t2[0:5]
    confirmdate = datetime.datetime(y, m, d, h, mi)
    gap = (confirmdate - orderdate).days*1440 + int((confirmdate - orderdate).seconds/60.0)
    return gap
# print(getordergap('2017/2/6  9:07', '2017/2/8  10:57'))   #2990
"""

"""
#判断月份(7,8,9月分别对应0,1,2)
def getmonth(arrival_str):
    t = time.strptime(arrival_str, "%Y/%m/%d")
    m = t[1]
    dict = {7: 0, 8: 1, 9: 2}
    if m in [7, 8, 9]:
        return dict[m]
    else:
        return -1
print(getmonth('2017/8/10'))
#判断下单时间是否为工作时间
def isworking(orderdate_str):
    t = time.strptime(orderdate_str, "%Y/%m/%d %H:%M")
    h = t[3]
    if h >= 18 or h <= 8:
        return False
    else:
        return True
print(isworking('2017/7/19 8:45'))

#获取两个日期的差，单位是天数(int)
def getdaygap(date1_str, date2_str):
    t1 = time.strptime(date1_str, "%Y/%m/%d")
    y, m, d = t1[0:3]
    date1 = datetime.date(y, m, d)
    t2 = time.strptime(date2_str, "%Y/%m/%d")
    y, m, d = t2[0:3]
    date2 = datetime.date(y, m, d)
    gap = (date2 - date1).days
    return gap
print(getdaygap('2017/2/4', '2017/6/7'))        #123
#判断是否为节假日
def isspecialday(date_str):
    specialdays = {'2017/2/11', '2017/2/14', '2017/4/1', '2017/5/30', '2017/8/28'}
    if date_str in specialdays:
        return True
    else:
        return False
print(isspecialday('2017/8/28'))        #True
#判断是否为周末
def isweekend(date_str):
    if getdayinfo(date_str)[1] == 5 or getdayinfo(date_str)[1] == 6:
        return True
    else:
        return False
print(isweekend('2018/7/30'))           #False
"""

# 携程海外酒店房态预测算法大赛，时间数据的初步处理
# https://www.kesci.com/home/competition/5b18a9d7fe8bc06aa3a937b5/content/3
"""
import pandas as pd
data = pd.read_csv('ord_train.csv')
arrival = data['arrival']
noroom = data['noroom']
"""

"""
import numpy as np
import pandas as pd
data = pd.read_csv('ord_train.csv')
y = data.loc[:, 'noroom'].as_matrix(columns=None)
x = data.loc[:, ['city', 'ordadvanceday', 'isvendor', 'hotelstar']].as_matrix(columns=None)
#线性回归
from sklearn.linear_model import LinearRegression
linR = LinearRegression()
linR.fit(x, y)
print(linR.coef_)                           #系数
print(linR.score(x, y))                     #相关系数R
print(np.mean((linR.predict(x) - y)**2))    #均方误差
#逻辑回归
from sklearn.linear_model.logistic import LogisticRegression
logR = LogisticRegression()
logR.fit(x, y)
print(logR.coef_)                           #系数
print(logR.score(x, y))                     #相关系数R
print(np.mean((logR.predict(x) - y)**2))    #均方误差
"""

"""
#处理入住时间，生成表
import numpy as np
import pandas as pd
data = pd.read_csv('ord_train.csv')
a = data.loc[:, 'arrival'].as_matrix(columns=None)
b = data.loc[:, 'etd'].as_matrix(columns=None)
c = data.loc[:, 'orderdate'].as_matrix(columns=None)
staydays = []
monthid = []
weekend =[]
working = []
for i in range(len(a)):
    staydays.append(getdaygap(a[i], b[i]))
    monthid.append(getmonth(a[i]))
    weekend.append(isweekend(a[i]))
    working.append(isworking(c[i]))
staydays = pd.Series(staydays)
monthid = pd.Series(monthid)
weekend = pd.Series(weekend)
working = pd.Series(working)
arrival_info = pd.DataFrame(list(zip(data['arrival'], monthid, weekend, working, data['ordadvanceday'], staydays)),
columns=['arrival', 'monthid', 'isweekend', 'isworking', 'ordadvanceday', 'staydays'])
arrival_info.to_csv('arrival_info.csv')
"""

"""
#建立酒店维护类别(渠道)和订单处理时间(确定时间-下单时间)的关系,这里用的是平均值
import numpy as np
import pandas as pd
data = pd.read_csv('ord_train.csv')
data = data.drop('zone', axis=1, inplace=False)
data = data.dropna()
b = data['orderdate'].as_matrix(columns=None)
c = data['confirmdate'].as_matrix(columns=None)
ordergap = []
for i in range(len(b)):
    ordergap.append(getordergap(b[i], c[i]))
ordergap = pd.Series(ordergap)
order_info = pd.DataFrame(list(zip(data['orderdate'], data['confirmdate'], ordergap, data['hotelbelongto'])),
columns=['orderdate', 'confirmdate', 'ordergap', 'hotelbelongto'])
# order_info.to_csv('order_info.csv')
gaps_belong = pd.DataFrame(list(zip(ordergap, data['hotelbelongto'])),
columns=['ordergap', 'hotelbelongto'])
gaps_SHT = gaps_belong[gaps_belong['hotelbelongto'] == 'SHT']
gap_SHT = np.mean(gaps_SHT['ordergap'])
gaps_HTL = gaps_belong[gaps_belong['hotelbelongto'] == 'HTL']
gap_HTL = np.mean(gaps_HTL['ordergap'])
gaps_PKG = gaps_belong[gaps_belong['hotelbelongto'] == 'PKG']
gap_PKG = np.mean(gaps_PKG['ordergap'])
gaps_HPP = gaps_belong[gaps_belong['hotelbelongto'] == 'HPP']
gap_HPP = np.mean(gaps_HPP['ordergap'])
print(gap_SHT)      #44.80749089379228
print(gap_HTL)      #28.772667360388485
print(gap_PKG)      #303.54705225269953
print(gap_HPP)      #0.0
"""

"""
#禁用科学计数法，对表中数值型数据做描述统计
import pandas as pd
data = pd.read_csv('ord_train.csv')
data_int = pd.DataFrame(list(zip(data['noroom'], data['ordadvanceday'],
                                 data['price'], data['commission'],
                                 data['isvendor'], data['hotelstar'], data['ordroomnum'],)),
columns=['noroom', 'ordadvanceday', 'price', 'commission', 'isvendor', 'hotelstar', 'ordroomnum'])
with pd.option_context('display.float_format', lambda x: '%.3f' % x):
    print(data_int.describe())

"""

"""
import pandas as pd
data = pd.read_csv('ord_train.csv', dtype=str)
con = data['confirmdate']
print(con.describe())       
# count              968358
# unique             268770
# top       2017/6/26 10:09
# freq                   30
"""

"""
#按欧式距离填充区号的缺失值
import numpy as np
import pandas as pd
data = pd.read_csv('ord_train.csv')
a = data['zone'].as_matrix(columns=None)
b = data['city'].as_matrix(columns=None)
c = data['country'].as_matrix(columns=None)
def euclid(acountry, acity, bcountry, bcity):
    return np.sqrt((bcountry-acountry)**2 + (bcity-acity)**2)
for i in len(a):
    if a[i] == 'NULL':
        for i in
"""
