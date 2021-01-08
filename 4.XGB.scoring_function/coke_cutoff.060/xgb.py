import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import sys

nFeat = 6

data_in = np.array(pd.read_csv(sys.argv[1]))
nTarget = int(sys.argv[2])
nData = int(sys.argv[3])

X = data_in[0:nData,0:nFeat]
Y = data_in[0:nData,nFeat+nTarget-1]

X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
Y_train, Y_test = train_test_split(Y, test_size=0.2, random_state=0)

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
print('Data: {}\tnData: {}\tnTarget: {}\topt d={}\topt n={}\tmin MAE: {:.4f}'.format(sys.argv[1], nData, nTarget, opt_d, opt_n, mae))

nTest = len(Y_test)

ndx = range(nTest)
prt=pd.DataFrame()
prt["id"] = ndx
prt["ref"] = Y_test
prt["xgb"] = Y_test_out
prt.to_csv("output.test.xgb.nData.%d.Target.%d.csv"%(nData,nTarget),index=False)
