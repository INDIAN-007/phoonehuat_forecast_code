# from utility import make_time_series_continous_
import numpy as np
import pandas as pd
from Croston_Algo import Croston_TSB
from pmdarima import auto_arima
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

from utility import fill_horizon_pred


# c1=meta_data['horizon']==6
# This method tAKES TWO LOCAL VARIABLES
# df that is the main data
# meta_data that is the map for dividing the data from into trainingset and test set
def get_sku_level_forecast(sku, current_date, df, meta_data, max_horizon):
    c2 = meta_data['sku'] == sku
    temp_meta_data = meta_data[c2]
    c1 = df['sku'] == sku
    sku_data = df[c1]
    min_date = sku_data['date_'].min()
    # sku_data=make_time_series_continous_(sku_data,min_date,current_date+pd.DateOffset(months=max_horizon))
    # sku_data['date_']=pd.to_datetime(sku_data['date'].astype(str).str[:4]+"-"+sku_data['date'].astype(str).str[4:6]+"-01")
    # sku_data['date_']=pd.to_datetime(sku_data['date'].astype(str).str[:4]+"-"+sku_data['date'].astype(str).str[4:6]+"-01")
    sku_data['sales'].fillna(0, inplace=True)

    temp_meta_data_ = temp_meta_data[["train_data_start", "loop_back_months_for_weights", 'train_data_end', "horizon",
                                      'sku']].drop_duplicates().reset_index(drop=True)
    temp_meta_data_['actual_demand'] = np.nan
    temp_meta_data_['arima'] = np.nan
    temp_meta_data_['simple_exp'] = np.nan
    temp_meta_data_['Holt_linear'] = np.nan
    temp_meta_data_['Holt_additive_damped'] = np.nan
    temp_meta_data_['croston'] = np.nan
    for i in range(temp_meta_data_.shape[0]):

        sku_meta_data = temp_meta_data_.iloc[i]
        horizon = sku_meta_data['horizon']
        train_cond = sku_data['date_'] <= sku_meta_data['train_data_end']
        test_cond_start = sku_data['date_'] > sku_meta_data['train_data_end']
        test_cond_end = sku_data['date_'] <= sku_meta_data['loop_back_months_for_weights']
        train_data = sku_data[train_cond]
        train_data.index = train_data['date']
        test_data = sku_data[test_cond_start & test_cond_end]
        test_data.index = test_data['date']
        train = train_data
        print("SKU META")
        # print(sku_meta_data)
        # print(sku_meta_data.dtypes)
        print(test_data, horizon)
        try:
            temp_meta_data_.loc[i, "actual_demand"] = test_data.iloc[horizon - 1]['sales']
        except:
            temp_meta_data_.loc[i, "actual_demand"] = 0
        # ARIMA MODEL
        try:
            # arima_model = auto_arima(train['sales'], start_p=0, d=None, start_q=0,
            #                          max_p=5, max_d=1, max_q=5, start_P=0, D=None, start_Q=0,
            #                          max_P=5, max_D=1, max_Q=5, m=4, seasonal=True,
            #                          error_action='ignore', trace=True,
            #                          supress_warnings=True, stepwise=True,
            #                          random_state=20, n_fits=10, information_criterion='aic',
            #                          scoring='mse', seasonal_test_args={"max_lag": 4})

            arima_model = auto_arima(train['sales'],
                                     seasonal=True,
                                     m=12,  # For monthly data with yearly seasonality
                                     trace=True,
                                     error_action='ignore',
                                     suppress_warnings=True,
                                     max_p=2, max_d=1, max_q=2,
                                     # Restricting the range of p, d, q
                                     max_P=2, max_D=1, max_Q=2,  # Restricting the range of seasonal p, d, q
                                     max_order=4,  # Setting a maximum order to limit the search space
                                     #    n_jobs=-1,
                                     stepwise=True)

            arima = arima_model.predict(n_periods=int(horizon)).reset_index(drop=True)[horizon - 1]
            temp_meta_data_.loc[i, "arima"] = arima if arima > 0 else 0
        except:
            temp_meta_data_.loc[i, "arima"] = np.nan

        # Simple Exponential Model
        try:
            ses_model = SimpleExpSmoothing(train['sales'].astype(int)).fit()
            pred_ses = ses_model.forecast(horizon).rename(r"$\alpha=%s$" % ses_model.model.params["smoothing_level"])
            simple_exp = pred_ses.values[horizon - 1]
            temp_meta_data_.loc[i, "simple_exp"] = simple_exp if simple_exp > 0 else 0
        except:
            temp_meta_data_.loc[i, "simple_exp"] = np.nan
        # Holts Linear
        try:
            fit1 = Holt(train['sales'].astype(int), initialization_method="estimated").fit(
                smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
            fcast1 = fit1.forecast(horizon).rename("Holt's linear trend")
            holt_linear = fcast1.values[horizon - 1]
            temp_meta_data_.loc[i, "Holt_linear"] = holt_linear if holt_linear > 0 else 0
        except:
            temp_meta_data_.loc[i, "Holt_linear"] = np.nan

        # Holts Additive Damped
        try:
            fit3 = Holt(train['sales'].astype(int), damped_trend=True, initialization_method="estimated").fit(
                smoothing_level=0.8, smoothing_trend=0.2)
            fcast3 = fit3.forecast(horizon).rename("Additive damped trend")
            holt_additive_damped = fcast3.values[horizon - 1]
            temp_meta_data_.loc[i, "Holt_additive_damped"] = holt_additive_damped if holt_additive_damped > 0 else 0
        except:
            temp_meta_data_.loc[i, "Holt_additive_damped"] = np.nan

        ########################### CROSTON FORECAST FOR INTERMITTENT DEMAND ############################
        try:
            cros = Croston_TSB(train['sales'].reset_index(drop=True), extra_periods=horizon)
            cros_forecast = cros['Forecast']
            croston = cros_forecast.values[horizon - 1]
            temp_meta_data_.loc[i, "croston"] = croston if croston > 0 else 0
        except:
            temp_meta_data_.loc[i, "croston"] = np.nan
        print(temp_meta_data_.loc[i, :], "Printing META DETAILS")
    return pd.merge(temp_meta_data, temp_meta_data_, on=["train_data_start", "loop_back_months_for_weights", \
                                                         'train_data_end', "horizon", 'sku'], how='left')


def get_sku_level_forecast_new(sku, current_date, df, meta_data, max_horizon):
    c2 = meta_data['sku'] == sku
    temp_meta_data = meta_data[c2]
    c1 = df['sku'] == sku
    sku_data = df[c1]
    min_date = sku_data['date_'].min()
    # sku_data=make_time_series_continous_(sku_data,min_date,current_date+pd.DateOffset(months=max_horizon))
    # sku_data['date_']=pd.to_datetime(sku_data['date'].astype(str).str[:4]+"-"+sku_data['date'].astype(str).str[4:6]+"-01")
    # sku_data['date_']=pd.to_datetime(sku_data['date'].astype(str).str[:4]+"-"+sku_data['date'].astype(str).str[4:6]+"-01")
    sku_data['sales'].fillna(0, inplace=True)

    for i in temp_meta_data['train_data_end'].unique():
        c = sku_data['date_'] <= i
        train_data = sku_data[c]
        c1 = sku_data['date_'] > i
        test_data = sku_data[c1]
        test_data = test_data.iloc[:max_horizon, :]
        #     display(train_data)
        #     display(test_data)

        try:
            arima_model = auto_arima(train_data['sales'],
                                     seasonal=True, m=12,  # For monthly data with yearly seasonality
                                     trace=True, error_action='ignore', suppress_warnings=True,
                                     max_p=2, max_d=1, max_q=2, max_P=2, max_D=1, max_Q=2,
                                     # Restricting the range of seasonal p, d, q
                                     max_order=4,  # Setting a maximum order to limit the search space
                                     stepwise=True)

            arima = arima_model.predict(n_periods=int(max_horizon)).reset_index(drop=True)
            fill_horizon_pred(temp_meta_data, i, 'arima', arima)
        except:
            pass

        try:
            ses_model = SimpleExpSmoothing(train_data['sales'].astype(int)).fit()
            pred_ses = ses_model.forecast(max_horizon).rename(
                r"$\alpha=%s$" % ses_model.model.params["smoothing_level"])
            simple_exp = pred_ses.values
            print(simple_exp)
            fill_horizon_pred(temp_meta_data, i, 'simple_exp', simple_exp)
        except:
            pass
        try:
            fit1 = Holt(train_data['sales'].astype(int), initialization_method="estimated").fit(
                smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
            fcast1 = fit1.forecast(max_horizon).rename("Holt's linear trend")
            holt_linear = fcast1.values
            fill_horizon_pred(temp_meta_data, i, 'Holt_linear', holt_linear)
        except:
            pass

        try:
            fit3 = Holt(train_data['sales'].astype(int), damped_trend=True, initialization_method="estimated").fit(
                smoothing_level=0.8, smoothing_trend=0.2)
            fcast3 = fit3.forecast(max_horizon).rename("Additive damped trend")
            holt_additive_damped = fcast3.values
            fill_horizon_pred(temp_meta_data, i, 'Holt_additive_damped', holt_additive_damped)
        except:
            pass

        try:
            cros = Croston_TSB(train_data['sales'].reset_index(drop=True), extra_periods=max_horizon)
            cros_forecast = cros['Forecast']
            croston = cros_forecast.values
            fill_horizon_pred(temp_meta_data, i, 'croston', croston)
        except:
            pass

    return temp_meta_data
