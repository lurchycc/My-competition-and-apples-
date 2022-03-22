# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)

#--------------------------------------------数据预处理------------------------------------------------------

#数据集加载
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


#处理离群点
train_df.drop(train_df[(train_df['月租金']>0.3)].index,inplace=True)


#--------------------------------------------特征工程------------------------------------------------------

#处理分类特征
def one_hot(df,col,nan_as_category=True):
    df = pd.get_dummies(df,columns=[col],dummy_na=nan_as_category)
    return df

#特征工程1:替换少数类 CV降低0.01
direction_dict = dict(df['房屋朝向'].value_counts())
minority_class = []
for key,value in direction_dict.items():
    if value < 4000:
        minority_class.append(key)

df['房屋朝向'] = df['房屋朝向'].replace(minority_class,'其他')

df = one_hot(df,'房屋朝向')

#特征工程2:特征提取
# 提取分类变量中对因变量影响较大的类 配合调参 线上有百分位的减小
df['区_6'] = (df['区'] == 6.0).astype(np.int64)
df['区_10'] = (df['区'] == 10.0).astype(np.int64)
df['区_12'] = (df['区'] == 12.0).astype(np.int64)

#提取连续变量中对应月租金方差较大的值 线上降低0.04,线下降低0.005
df['位置'] = df['位置'].fillna(9999).astype(np.int64)
df['位置_is_78'] = (df['位置'] == 78).astype(np.int64)
df['位置_is_43'] = (df['位置'] == 43).astype(np.int64)
df['位置_is_137'] = (df['位置'] == 137).astype(np.int64)
df['位置_is_102'] = (df['位置'] == 102).astype(np.int64)
df['位置_is_71'] = (df['位置'] == 71).astype(np.int64)
df['位置'] = df['位置'].replace({9999,np.nan})

#特征工程3:加减乘除 构造新的特征

# 线上减小0.07,线下减少0.04
df['总房间数量'] = df['卧室数量'] + df['厅的数量'] + df['卫的数量']
df['每个房间面积'] = df['房屋面积'] / df['总房间数量']
#线上减小0.001
df['楼层*总楼层'] = df['楼层']*df['总楼层']


#--------------------------------------------建模与评估------------------------------------------------------

params = {
    'objective': 'regression',
    'boosting_type': 'gbdt',
    'nthread':4,
    'learning_rate':0.1, #0.14
    'colsample_bytree':0.86, #0.95
    'subsample': 0.8,
    'max_depth':6,
    #"lambda_l1": 0.1,
    "lambda_l2": 0.1,#0.125
    'seed': 0,
    'verbose': -1,
    'metric': 'l2_root',
    #'min_child_weight': 10,
    #'min_split_gain': 0.1,
}

num_round = 5000 #30000

train = df[df['月租金'].notnull()]
test = df[df['月租金'].isnull()]
feats = [feat for feat in df.columns if feat not in ['月租金', 'id', '时间','小区房屋出租数量']]


#线下交叉验证
def kfold_lgbm_cv(df,params,num_round):

    folds = KFold(n_splits=5,shuffle=True,random_state=2018)

    oof_preds = np.zeros(train.shape[0])

    for n_fold, (train_idx,valid_idx) in enumerate(folds.split(train[feats], train['月租金'])):


            dtrain = lgb.Dataset(data=train[feats].iloc[train_idx],
                                 label=train['月租金'].iloc[train_idx],
                                 free_raw_data=False, silent=True)

            dvalid = lgb.Dataset(data=train[feats].iloc[valid_idx],
                                 label=train['月租金'].iloc[valid_idx],
                                 free_raw_data=False, silent=True)


            model = lgb.train(
                    params=params,
                    train_set=dtrain,
                    valid_sets=[dtrain, dvalid],
                    num_boost_round=num_round,
                    early_stopping_rounds=50,
                    #verbose_eval=False,
            )


            oof_preds[valid_idx] = model.predict(dvalid.data)
            print('Fold %2d RMSE : %.6f' % (n_fold + 1,np.sqrt(mean_squared_error(dvalid.label,oof_preds[valid_idx]))))

    print('Validation RMSE score %.6f' % np.sqrt(mean_squared_error(train['月租金'],oof_preds)))



kfold_lgbm_cv(df,params,num_round)

X_train = lgb.Dataset(data=train[feats],
                     label=train['月租金'],
                     free_raw_data=False, silent=True)

reg = lgb.train(
    params=params,
    train_set=X_train,
    num_boost_round=num_round,
)


feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = feats
feature_importance_df["importance"] = reg.feature_importance(importance_type='gain')

sub_pred = reg.predict(test[feats])

sub_df = pd.DataFrame({'id': test['id'].copy(),
                       'price': sub_pred})
sub_df['id'] = sub_df['id'].astype(np.int64)
#反常项处理
#sub_df.loc[sub_df['id']==13919,'price'] = 67.172397
sub_df[['id', 'price']].to_csv('Log_baseline.csv', index=False)
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
feature_importance_df.to_csv('feature_importance.csv')























