import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import preprocessing
from matplotlib import pyplot as plt

# 编码
def encode(data_ready,name,values):
    df_temp = data_ready[name]
    new_values = []
    for value in df_temp.values:
        for i in range(len(values)):
            key = values[i]
            if value == key:
                new_values += [i]
    data_ready[name] = new_values
    return data_ready

# kmeans模型
def kmeans(n_clusters,train_x):
    kmeans = KMeans(n_clusters)
    kmeans.fit(train_x)
    predict_y = kmeans.predict(train_x)
    score = kmeans.score(train_x)
    return predict_y, score

# 初始设定
pd.set_option('display.max_columns', 10,'display.max_rows', 300,'display.width', 200)

# 第一步：数据加载
data_raw = pd.read_csv('./car_price.csv')
data_ready = data_raw.copy(deep=True)
# 第二步：数据预处理20

# 处理fueltype
name = 'fueltype'
value1= 'diesel'
value2 = 'gas'
values = [value1,value2]
data_ready = encode(data_ready,name,values)

# 处理aspiration
name = 'aspiration'
value1 = 'std'
value2 = 'turbo'
values = [value1,value2]
data_ready = encode(data_ready,name,values)

# 处理doornumber
name = 'doornumber'
value1 = 'two'
value2 = 'four'
values = [value1,value2]
data_ready = encode(data_ready,name,values)

# 处理carbody
name = 'carbody'
values = ['convertible','hardtop','hatchback','sedan','wagon']
data_ready = encode(data_ready,name,values)

# 处理drivewheel
name = 'drivewheel'
values = ['fwd','4wd','rwd']
data_ready = encode(data_ready,name,values)

# 处理enginelocation
name = 'enginelocation'
values = ['front', 'rear']
data_ready = encode(data_ready,name,values)

# 处理enginetype
name = 'enginetype'
values = ['dohc', 'dohcv', 'l','ohc', 'ohcf', 'ohcv', 'rotor']
data_ready = encode(data_ready,name,values)

# 处理cylindernumber
name = 'cylindernumber'
values = ['two', 'three', 'four', 'five', 'six', 'eight', 'twelve']
data_ready = encode(data_ready,name,values)

# 处理fuelsystem
name = 'fuelsystem'
values = ['1bbl', '2bbl', '4bbl', 'idi', 'mfi', 'mpfi', 'spdi', 'spfi']
data_ready = encode(data_ready,name,values)

# index更新
data_ready.index = data_ready['CarName']

# 删除不分析的数据列
data_ready.drop(['car_ID', 'CarName'],axis=1, inplace=True)

# 使用KNN聚类
# 第三步： 归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
normal_x = min_max_scaler.fit_transform(data_ready)

# 第四步：调整权重
weight = data_ready.shape[1]-1
train_x = normal_x
m = len(train_x)
for i in range(m):
    train_x[i,-1] = train_x[i,-1]*weight

# 第五步：手肘法，分析不同k值的聚合误差
range = range(2,20)
score = []
for n_clusters in range:
    # kmeans聚类分析
    score = score+[kmeans(n_clusters,train_x)[1]]
plt.xlabel('K')
plt.ylabel('score')
plt.plot(range, score, 'o-')
plt.show()

# 第六步：选择合适的k值进行聚合分析
# kmeans聚类分析
n_clusters = int(input('请输入需要聚合分类的K值：'))
# n_clusters = 20
predict_y = kmeans(n_clusters,train_x)[0]
df_result = data_ready
df_result['聚类结果'] = predict_y
df_result.sort_values(by='聚类结果',inplace=True,ascending=False)

# 第七步：竞品车型
df_comptition_cars = pd.read_csv('./car_price.csv')
df_comptition_cars['聚类结果'] = predict_y

# 所有车名
car_names = df_comptition_cars['CarName']
# VW车名
car_names_vw = [value for value in car_names if ('vw' in value) or ('volkswagen' in value)]

df_comptition_cars.index=df_comptition_cars['CarName']
df_vw_cars =df_comptition_cars.loc[car_names_vw]
group_no = list(set(df_vw_cars['聚类结果']))
df_comptition_cars.drop(['CarName'],axis=1, inplace=True)

# 第八步：数据呈现
for value in group_no:
    print('\n')
    print('VW竞品车聚类： %s 类' % value)
    print(df_comptition_cars[df_comptition_cars['聚类结果']==value])

# # 使用层次聚类
# # 第一步：数据规范化
# min_max_scaler = preprocessing.MinMaxScaler()
# data_ready = min_max_scaler.fit_transform(data_ready)
#
# # 第二步：使用层次聚类
# model = AgglomerativeClustering(n_clusters=3, linkage='ward')
# y = model.fit_predict(data_ready)
#
# # 第三步：合并聚类结果，插入到原数据中
# result = pd.concat((data_raw, pd.DataFrame(y)), axis=1)
# result.rename({0:u'聚类结果'}, axis=1, inplace=True)