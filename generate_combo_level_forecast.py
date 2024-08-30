from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from Croston_Algo import Croston_TSB
import pandas as pd
import numpy as np
from utility import make_date_continous, fill_horizon_pred
from tqdm import tqdm
from pmdarima import auto_arima


def generate_combo_level_forecast(heirarchy_list, combo_data, combo_meta_data, max_horizon, current_date):

    combo_level_forecast = pd.DataFrame()
    for combi in heirarchy_list:
        sales_combo = f"sales_{'_'.join(combi)}"
        combo_col_name_ = f"combo_name_{'_'.join(combi)}"
        combo_data_ = combo_data[[combo_col_name_, 'year_month', sales_combo]]
        combo_name_level_forecast = pd.DataFrame()

        for combo_name_ in tqdm(combo_data_[combo_col_name_].unique()):
            combo_name_level_data = pd.DataFrame()
            print(combo_name_)
            combo_temp_condition = combo_data_[combo_col_name_] == combo_name_
            temp_combo_data = combo_data_[combo_temp_condition]

            temp_combo_data = temp_combo_data.groupby([combo_col_name_, 'year_month'])[sales_combo].max().reset_index()
            print(temp_combo_data.tail(20))
            temp_combo_data = make_date_continous(temp_combo_data, combo_col_name_, 'year_month',
                                                  current_date + pd.DateOffset(months=max_horizon))
            temp_combo_data['date'] = pd.to_datetime(temp_combo_data['year_month'] + "-01")
            cmd_condition = combo_meta_data['combo_name'] == combo_name_
            temp_combo_meta_data = combo_meta_data[cmd_condition]
            temp_combo_meta_data_ = temp_combo_meta_data[
                ['loop_back_months_for_weights', 'train_data_end', 'train_data_start', 'horizon']].drop_duplicates()
            temp_combo_meta_data_['actual_demand'] = np.nan
            temp_combo_meta_data_['arima'] = np.nan
            temp_combo_meta_data_['simple_exp'] = np.nan
            temp_combo_meta_data_['Holt_linear'] = np.nan
            temp_combo_meta_data_['Holt_additive_damped'] = np.nan
            temp_combo_meta_data_['croston'] = np.nan
            temp_combo_meta_data_['model_status'] = np.nan

            for i in range(temp_combo_meta_data_.shape[0]):
                meta_details = temp_combo_meta_data_.iloc[i, :]
                train_end = temp_combo_data['date'] <= meta_details['train_data_end']
                test_start = temp_combo_data['date'] > meta_details['train_data_end']
                test_end = temp_combo_data['date'] <= meta_details['loop_back_months_for_weights']
                train_data = temp_combo_data[train_end]
                test_data = temp_combo_data[test_start & test_end]

                print(train_data.tail(20))
                print(train_data['year_month'])
                print(list(train_data[sales_combo]))
                try:
                    meta_details['actual_demand'] = test_data[sales_combo].values[-1]
                except:
                    meta_details['actual_demand'] = np.nan
                horizon = meta_details['horizon']
                try:

                    arima_model = auto_arima(train_data[sales_combo].fillna(0),
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
                    meta_details["arima"] = arima if arima > 0 else 0

                except:
                    meta_details["arima"] = np.nan
                try:
                    ses_model = SimpleExpSmoothing(train_data[sales_combo].astype(int)).fit()
                    pred_ses = ses_model.forecast(horizon).rename(
                        r"$\alpha=%s$" % ses_model.model.params["smoothing_level"])
                    simple_exp = pred_ses.values[horizon - 1]
                    meta_details["simple_exp"] = simple_exp if simple_exp > 0 else 0
                except:
                    meta_details["simple_exp"] = np.nan

                try:
                    fit1 = Holt(train_data[sales_combo].astype(int), initialization_method="estimated").fit(
                        smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
                    fcast1 = fit1.forecast(horizon).rename("Holt's linear trend")
                    holt_linear = fcast1.values[horizon - 1]
                    meta_details["Holt_linear"] = holt_linear if holt_linear > 0 else 0
                except:
                    meta_details["Holt_linear"] = np.nan

                try:
                    fit3 = Holt(train_data[sales_combo].astype(int), damped_trend=True,
                                initialization_method="estimated").fit(
                        smoothing_level=0.8, smoothing_trend=0.2)
                    fcast3 = fit3.forecast(horizon).rename("Additive damped trend")
                    holt_additive_damped = fcast3.values[horizon - 1]
                    meta_details["Holt_additive_damped"] = holt_additive_damped if holt_additive_damped > 0 else 0
                except:
                    meta_details["Holt_additive_damped"] = np.nan

                try:
                    cros = Croston_TSB(train_data[sales_combo].reset_index(drop=True), extra_periods=horizon)
                    cros_forecast = cros['Forecast']
                    croston = cros_forecast.values[horizon - 1]
                    meta_details["croston"] = croston if croston > 0 else 0
                except:
                    meta_details["croston"] = np.nan
                print(meta_details)
                temp_combo_meta_data_.iloc[i, :] = meta_details

            temp_combo_meta_data = pd.merge(temp_combo_meta_data, temp_combo_meta_data_,
                                            on=['loop_back_months_for_weights', 'train_data_end', 'train_data_start',
                                                'horizon']
                                            , how='left')
            combo_name_level_forecast = pd.concat([combo_name_level_forecast, temp_combo_meta_data], axis=0)
        combo_level_forecast = pd.concat([combo_level_forecast, combo_name_level_forecast], axis=0)

    return combo_level_forecast


def generate_combo_level_forecast_new(heirarchy_list, combo_data, combo_meta_data, max_horizon, current_date):
    combo_level_forecast = pd.DataFrame()
    for combi in heirarchy_list:
        sales_combo = f"sales_{'_'.join(combi)}"
        combo_col_name_ = f"combo_name_{'_'.join(combi)}"
        combo_data_ = combo_data[[combo_col_name_, 'year_month', sales_combo]]
        combo_name_level_forecast = pd.DataFrame()

        for combo_name_ in tqdm(combo_data_[combo_col_name_].unique()[:5]):
            combo_name_level_data = pd.DataFrame()
            print(combo_name_)
            combo_temp_condition = combo_data_[combo_col_name_] == combo_name_
            temp_combo_data = combo_data_[combo_temp_condition]

            temp_combo_data = temp_combo_data.groupby([combo_col_name_, 'year_month'])[sales_combo].max().reset_index()
            print(temp_combo_data.tail(20))
            temp_combo_data = make_date_continous(temp_combo_data, combo_col_name_, 'year_month',
                                                  current_date + pd.DateOffset(months=max_horizon))
            temp_combo_data['date'] = pd.to_datetime(temp_combo_data['year_month'] + "-01")
            cmd_condition = combo_meta_data['combo_name'] == combo_name_
            temp_combo_meta_data = combo_meta_data[cmd_condition]
            temp_combo_meta_data_ = temp_combo_meta_data[
                ['loop_back_months_for_weights', 'train_data_end', 'train_data_start', 'horizon']].drop_duplicates()
            temp_combo_meta_data_['actual_demand'] = np.nan
            temp_combo_meta_data_['arima'] = np.nan
            temp_combo_meta_data_['simple_exp'] = np.nan
            temp_combo_meta_data_['Holt_linear'] = np.nan
            temp_combo_meta_data_['Holt_additive_damped'] = np.nan
            temp_combo_meta_data_['croston'] = np.nan
            temp_combo_meta_data_['model_status'] = np.nan

            for i in temp_combo_meta_data_['train_data_end'].unique():
                print(i)

                c = temp_combo_meta_data_['train_data_end'] == i
                meta_details = temp_combo_meta_data_[c]
                print("Printing Meta _details")
                train_end = temp_combo_data['date'] <= i
                test_start = temp_combo_data['date'] > i
                train_data = temp_combo_data[train_end]
                test_data = temp_combo_data[test_start].iloc[:max_horizon, :]


                print(train_data.tail(20))
                print(train_data['year_month'])
                print(list(train_data[sales_combo]))
                try:
                    meta_details['actual_demand'] = test_data[sales_combo].values[-1]
                except:
                    meta_details['actual_demand'] = np.nan
                try:

                    arima_model = auto_arima(train_data[sales_combo].fillna(0),
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

                    arima = arima_model.predict(n_periods=int(max_horizon)).reset_index(drop=True)
                    fill_horizon_pred(temp_combo_meta_data_, i, 'arima', arima)

                except:
                    pass
                try:
                    ses_model = SimpleExpSmoothing(train_data[sales_combo].astype(int)).fit()
                    pred_ses = ses_model.forecast(max_horizon).rename(
                        r"$\alpha=%s$" % ses_model.model.params["smoothing_level"])
                    simple_exp = pred_ses.values
                    # meta_details["simple_exp"] = simple_exp if simple_exp > 0 else 0
                    fill_horizon_pred(temp_combo_meta_data_, i, 'simple_exp', simple_exp)
                except:
                    # meta_details["simple_exp"] = np.nan
                    pass

                try:
                    fit1 = Holt(train_data[sales_combo].astype(int), initialization_method="estimated").fit(
                        smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
                    fcast1 = fit1.forecast(max_horizon).rename("Holt's linear trend")
                    holt_linear = fcast1.values
                    fill_horizon_pred(temp_combo_meta_data_, i, 'Holt_linear', holt_linear)
                except:
                    # meta_details["Holt_linear"] = np.nan
                    pass

                try:
                    fit3 = Holt(train_data[sales_combo].astype(int), damped_trend=True,
                                initialization_method="estimated").fit(
                        smoothing_level=0.8, smoothing_trend=0.2)
                    fcast3 = fit3.forecast(max_horizon).rename("Additive damped trend")
                    holt_additive_damped = fcast3.values
                    # meta_details["Holt_additive_damped"] = holt_additive_damped if holt_additive_damped > 0 else 0
                    fill_horizon_pred(temp_combo_meta_data_, i, 'Holt_additive_damped', holt_additive_damped)
                except:
                    # meta_details["Holt_additive_damped"] = np.nan
                    pass

                try:
                    cros = Croston_TSB(train_data[sales_combo].reset_index(drop=True), extra_periods=max_horizon)
                    cros_forecast = cros['Forecast']
                    croston = cros_forecast.values
                    fill_horizon_pred(temp_combo_meta_data_, i, 'croston', croston)
                except:
                    # meta_details["croston"] = np.nan
                    pass

            temp_combo_meta_data = pd.merge(temp_combo_meta_data, temp_combo_meta_data_,
                                            on=['loop_back_months_for_weights', 'train_data_end', 'train_data_start',
                                                'horizon']
                                            , how='left')
            combo_name_level_forecast = pd.concat([combo_name_level_forecast, temp_combo_meta_data], axis=0)
        combo_level_forecast = pd.concat([combo_level_forecast, combo_name_level_forecast], axis=0)


    return combo_level_forecast

