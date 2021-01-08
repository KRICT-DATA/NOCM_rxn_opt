import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

import sys

nFeat = 6
sol_data = np.array(pd.read_csv(sys.argv[1]))
nData = int(sys.argv[2])

X_sol = sol_data[:,0:nFeat]

for nTarget in range(1,7):
    df_in = np.array(pd.read_csv('data_renorm.csv'))
    data_in = df_in[:nData,:nFeat+nTarget]
    
    seed = 168
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
