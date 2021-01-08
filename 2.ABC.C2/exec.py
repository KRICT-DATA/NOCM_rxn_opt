import numpy
import pandas
import joblib
from artificial_bee_colony import ABC

predictor = joblib.load('predictor.sav')

def pnt_func(x):
    pnt = 0

    for i in range(0, x.shape[1]):
        if x[0, i] < -1.0:
            pnt -= 1

        if x[0, i] > 1.0:
            pnt -= 1

    return pnt

# heuristic search
num_feats = 6
lbs = numpy.ones([num_feats])*(-1.0)
ubs = numpy.ones([num_feats])*(1.0)
opt = ABC(num_feats, predictor.predict, lbs, ubs, opt_type='max', lim_trial=0.1, pnt_func=pnt_func)
sol, val = opt.run(1000)

numpy.savetxt('sol.csv', sol.reshape(1, -1), delimiter=',', header='pressure,max_T,flow_rate,H2_per_CH4_n_Ar,reacter_len,reacter_dia')
#numpy.savetxt('sol.csv', sol, delimiter=',')

print(sol)
print(predictor.predict(sol.reshape(1,-1)))
