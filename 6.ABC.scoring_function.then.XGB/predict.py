import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split

import sys

nFeat = 6
sol_data = np.array(pd.read_csv(sys.argv[1]))
nData = int(sys.argv[2])

X_sol = sol_data[:,0:nFeat]

for nTarget in range(1,7):
    df_in = np.array(pd.read_csv('data_renorm.csv'))
    data_in = df_in[:nData,:nFeat+nTarget]
    
    train_ratio = 0.8

    nTrain = int(nData*train_ratio)
    nTest = nData - nTrain

    X = data_in[0:nData,0:nFeat]
    Y = data_in[0:nData,nFeat+nTarget-1]

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=0)
    Y_train, Y_test = train_test_split(Y, test_size=0.2, random_state=0)

    opt_d = 5
    opt_n = 300

    # save predictor
    predictor = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n, subsample=0.8, verbosity=0)
    predictor.fit(X_train, Y_train, eval_metric='mae', eval_set=[(X_test, Y_test)])
    joblib.dump(predictor, "predictor.%d.%d.sav"%(nData, nTarget))

results = []

for target in range(1,7):
    predictor = joblib.load("predictor.%d.%d.sav"%(nData, target))
    
    results.append((predictor.predict(X_sol)))

results = np.asarray(results)
np.savetxt("results.%d.csv"%nData, results.reshape(1, -1), delimiter=',', header='Conv,C2H4,C2,Coke,Y_C2,Y_C2H4', comments='')
