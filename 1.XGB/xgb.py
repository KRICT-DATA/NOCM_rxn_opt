import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

import sys

nFeat = 6
nTarget = int(sys.argv[1])
nData = int(sys.argv[2])
seed = 168

df_in = np.array(pd.read_csv('data_renorm.csv'))
data_in = df_in[:nData,:nFeat+nTarget]
np.random.seed(seed)
np.random.shuffle(data_in)

train_ratio = 0.8

nTrain = int(nData*train_ratio)
nTest = nData - nTrain

X_train = data_in[:nTrain,:nFeat]
Y_train = data_in[:nTrain,nFeat+nTarget-1]

X_test = data_in[nTrain:,:nFeat]
Y_test = data_in[nTrain:,nFeat+nTarget-1]

opt_d = 5
opt_n = 300

# build predictor
test_err = 0.0
predictor = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n, subsample=0.8, verbosity=0)
predictor.fit(X_train, Y_train, eval_metric='mae', eval_set=[(X_test, Y_test)])
Y_test_out = predictor.predict(X_test)
test_err = np.mean(np.abs(Y_test_out - Y_test))
 
joblib.dump(predictor, "predictor.%d.%d.sav"%(nData, nTarget))

predictor = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n, subsample=0.8)
predictor.fit(X_train, Y_train, eval_metric='mae', eval_set=[(X_test, Y_test)])
Y_test_out = predictor.predict(X_test)

mae = np.mean(np.abs(Y_test_out - Y_test))
print('nData: {}\tnTarget: {}\topt d={}\topt n={}\tmin MAE: {:.4f}'.format(nData, nTarget, opt_d, opt_n, mae))

nTest = len(Y_test)

ndx = range(nTest)
prt=pd.DataFrame()
prt["id"] = ndx
prt["ref"] = Y_test
prt["xgb"] = Y_test_out
prt.to_csv("output.test.xgb.nData.%d.Target.%d.csv"%(nData,nTarget),index=False)
