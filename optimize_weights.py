# Finds the Mean Absolute Percentage Error
# number_of_algo=5
from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error
import scipy.optimize as opt
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def objective(xy,df,model_col):
    df["final_pred"]=df[model_col[:-1]].mul(xy).sum(axis=1)
    # print(df['actual_demand'])
    # print(df['final_pred'])
    rmse=sqrt(mean_squared_error(df['actual_demand'], df['final_pred']))
    return rmse

def optimise_weights(df,model_col):
    xy_start=[0]*len(model_col[:-1])
    bounds=tuple([(0,1)]*len(model_col[:-1]))

    y="+".join([f'xy[{i}]' for i in range(len(model_col[:-1]))])+'- 1'
    cons=({'type':'eq',"fun":lambda xy : eval(y)})
    result = opt.minimize(fun=objective, x0=xy_start, args=(df,model_col), options={'disp':True},
                          constraints=cons, bounds=bounds)
    df["final_pred"]=df[model_col[:-1]].mul(result.x).sum(axis=1)
    return df,result.x
