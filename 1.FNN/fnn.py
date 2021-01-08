from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import tensorflow as tf

import sys

seed = 168
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

#target: Conv, C2H4, C2, Coke, C2 yield, C2H4 yield

nTarget = int(sys.argv[2])
nData = int(sys.argv[3])

##
df_in = np.array(pd.read_csv(sys.argv[1]))

nFeat = 6
data_in = df_in[:nData,:nFeat+nTarget]
np.random.shuffle(data_in)

train_ratio = 0.8

nTrain = int(nData*train_ratio)
nTest = nData - nTrain

X_train = data_in[:nTrain,:nFeat]
Y_train = data_in[:nTrain,nFeat+nTarget-1]

X_test = data_in[nTrain:,:nFeat]
Y_test = data_in[nTrain:,nFeat+nTarget-1]

Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1000)

opt_active = 'sigmoid'
opt_nnode = 60
opt_optimizer = 'adadelta'

model = Sequential()
model.add(Dense(opt_nnode, input_dim=nFeat, activation=opt_active))
model.add(Dense(opt_nnode, activation=opt_active))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=opt_optimizer)
model.fit(X_train, Y_train, epochs=10000, batch_size=32, verbose=0, validation_split=0.1, callbacks=[early_stopping_callback])

model.save_weights("predictor.%d.%d.h5"%(nData, nTarget))

Y_test_pred = model.predict(X_test)

ndx = range(nTest)
prt=pd.DataFrame()
prt["id"] = ndx
prt["ref"] = Y_test
prt["fnn"] = Y_test_pred
prt.to_csv("output.test.fnn.%d.csv"%nTarget,index=False)

mae = np.mean(np.abs(Y_test_pred - Y_test))
print ("Target: %3d, nData: %4d, nNode: %3d, Active fnts.: %s, %s, Opt.: %s, MAE: %lf"%(nTarget, nData, opt_nnode, opt_active, 'linear', opt_optimizer, mae))
