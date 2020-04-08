from statsmodels.stats.outliers_influence import variance_inflation_factor
from toad.transform import WOETransformer, Combiner
import A_Feature, B_Transformer, C_Select, D_model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import toad, time

start = time.time()

d1 = pd.read_csv('LoanStats_2017Q1.csv', header=1, parse_dates=['earliest_cr_line'], low_memory=False)[:-2]
d2 = pd.read_csv('LoanStats_2017Q2.csv', header=1, parse_dates=['earliest_cr_line'], low_memory=False)[:-2]
d3 = pd.read_csv('LoanStats_2017Q3.csv', header=1, parse_dates=['earliest_cr_line'], low_memory=False)[:-2]
train = pd.concat([d1, d2], join='inner')
test = d3
print('数据加载完成')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 缺失\清洗

print('train处理中'.center(20, '—'))
train = A_Feature.feature(train)
print('test处理中'.center(20, '—'))
test = A_Feature.feature(test)
print('当前特征数量为:{}'.format(train.shape[1]), '\n')

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 特征加工

print('train特征加工'.center(20, '—'))
train = B_Transformer.fea_engn(train)
print('test特征加工'.center(20, '—'))
test = B_Transformer.fea_engn(test)
print('当前特征数量为:{}'.format(train.shape[1]), '\n')

# 分箱
features = train.columns.tolist()
features.remove('loan_status')
X_train, X_test, y_train, y_test = train[features], test[features], train['loan_status'], test['loan_status']
print('分箱处理中'.center(20, '—'))
X_train, X_test, y_train, y_test = B_Transformer.Combiner_Trans(X_train, X_test, y_train, y_test, category=True)
print('当前特征数量为:{}'.format(X_train.shape[1]), '\n')

train_copy = X_train.copy()
test_copy = X_test.copy()

# WOE
print('WOE'.center(20, '—'))
X_train, X_test, y_train, y_test = B_Transformer.WOE_Trans(X_train, X_test, y_train, y_test)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 特征筛选
print('特征筛选'.center(20, '—'))
# VIF = pd.DataFrame()
# VIF['VIF Factor'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
# VIF['feature'] = X_train.columns
# VIF.to_csv('VIF.csv')
X_train, X_test, y_train, y_test = C_Select.Select_feature(X_train, X_test, y_train, y_test, iv=0.025)
print('参与建模特征：\n', X_train.columns.tolist())

# 建模
print('建模'.center(20, '—'))
y_train_pred, y_train_prob, y_test_pred, y_test_prob, col_n = D_model.Model_train(X_train, X_test, y_train, y_test, n=5)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# 报告

print('报告'.center(20, '—'))
# 查看变量分布图
X_train['split'] = 'train'
X_test['split'] = 'test'
data = pd.concat([X_train, X_test], axis=0, join='inner')
a = pd.concat([y_train, y_test], axis=0, join='inner')
b = pd.concat([data, a], axis=1)
b.to_csv('data/data.csv')

# for column in col_n:
#     g = sns.kdeplot(data[column][(data['split'] == 'train')], color='Red', shade=True)
#     g = sns.kdeplot(data[column][(data['split'] == 'test')], color='Gray', shade=True, ax=g)
#     g.set_xlabel(column)
#     g.set_ylabel('Ratio')
#     g = g.legend(['train', 'test'])
#     plt.show()

# EDA报告，参与模型训练的变量情况
EDA = toad.detect(X_train[col_n])
EDA.to_csv('EDA.csv')
print('EDA done!')

# col = train_copy[col_n].columns[train_copy[col_n].dtypes != 'object']
# combiner = Combiner()
# WOE = WOETransformer()
# combiner.fit(X_train[col], y=y_train, method='chi', min_samples=0.1, empty_separate=False)
# X_train[col] = combiner.transform(X_train[col], labels=True)
# X_test[col] = combiner.transform(X_test[col], labels=True)
# # X_train[col] = WOE.fit_transform(X_train[col], y=y_train)
# # X_test[col] = WOE.fit_transform(X_test[col], y=y_test)
# print('Num_WOE done!')


# PSI报告，参与模型构建的特征稳定性
PSI = toad.metrics.PSI(X_train[col_n], X_test[col_n])
PSI.to_csv('PSI.csv')
print('PSI done!')

# KS报告
KS_train = toad.metrics.KS_bucket(y_train_prob, y_train, bucket=50, method='step')
KS_test = toad.metrics.KS_bucket(y_test_prob, y_test, bucket=50, method='step')
KS_train.to_csv('KS_train.csv')
KS_test.to_csv('KS_test.csv')
print('KS done!')

end = time.time()
print('总耗时：{:.2f}'.format((end - start)))
