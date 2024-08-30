import numpy as np
import pandas as pd
from tqdm import tqdm

from optimize_weights import optimise_weights
from static_variables import heirarchy_list
from utility import replace_arima_on_peak


def optimize_sku_combo_forecast(combo_wise_demand_forecast, meta_data, combo_data, sku_wise_demand_forecast,
                                output_file_path,peak_data):
    model_col = ['arima', 'simple_exp',
                 # "Holt_linear", "Holt_additive_damped",
                 # 'croston',
                 'actual_demand']

    h2_models = combo_wise_demand_forecast.copy()
    unique_horizon = h2_models.horizon.unique()
    unique_combo_name = h2_models['combo_name'].unique()
    weights_forecast_ = pd.DataFrame()
    for i in unique_horizon:
        for j in tqdm(unique_combo_name):
            c1 = h2_models['horizon'] == i
            c2 = h2_models['combo_name'] == j
            temp_data = h2_models[c1 & c2].fillna(0)
            # clear_output()
            for unique_forecast_month in temp_data['forecast_month'].unique():
                c4 = temp_data['forecast_month'] != temp_data['loop_back_months_for_weights']
                # print(unique_forecast_month)
                c5 = temp_data['forecast_month'] == unique_forecast_month
                d, fore_cast_ratio = optimise_weights(temp_data[c5], model_col)
                d = temp_data[c5]
                d["final_pred"] = d[model_col[:-1]].mul(fore_cast_ratio).sum(axis=1)
                weights_forecast_ = pd.concat([weights_forecast_, d], axis=0)

    weights_forecast_.to_csv(output_file_path + "weighted_forecast.csv", index=False, sep=',')

    ## ATTACHING COMBO DETAILS TO META DATA
    combo_columns = [f"combo_name_{'_'.join(i)}" for i in heirarchy_list]
    perc_columns = [f"perc_sales_{'_'.join(i)}" for i in heirarchy_list]
    meta_data['year_month'] = meta_data['loop_back_months_for_weights'].astype(str).str[:-3]
    meta_data_1 = pd.merge(meta_data, combo_data[['sku', 'year_month'] + combo_columns + perc_columns],
                           on=['sku', 'year_month'], how='left')

    ### Correcting minor missing errors to combo matching and filling missing values
    combo_columns = [f"combo_name_{'_'.join(i)}" for i in heirarchy_list]
    perc_columns = [f"perc_sales_{'_'.join(i)}" for i in heirarchy_list]
    for j, i in enumerate(combo_columns):
        print(i)
        meta_data_1 = pd.merge(meta_data_1.drop(i, axis=1), combo_data[['sku', i]].drop_duplicates())
        print(perc_columns[j])
        meta_data_1[perc_columns[j]].fillna(0, inplace=True)

    weights_forecast_['forecast_month'] = pd.to_datetime(weights_forecast_['forecast_month'])
    weights_forecast_['loop_back_months_for_weights'] = pd.to_datetime(
        weights_forecast_['loop_back_months_for_weights'])

    ## Attaching prediction values for meta data
    meta_data_2 = meta_data_1.copy()
    combo_columns = [f"combo_name_{'_'.join(i)}" for i in heirarchy_list]
    combo_name = ['_'.join(i) for i in heirarchy_list]
    perc_columns = [f"perc_sales_{'_'.join(i)}" for i in heirarchy_list]
    weights_forecast_['year_month'] = weights_forecast_['loop_back_months_for_weights'].astype(str).str[:-3]
    print(meta_data_2.head())
    meta_data_2['loop_back_months_for_weights'] = pd.to_datetime(meta_data_2['loop_back_months_for_weights'])
    meta_data_2['forecast_month'] = pd.to_datetime(meta_data_2['forecast_month'])

    for j, i in enumerate(combo_columns):
        #     print(i)
        #     print(combo_name[j])
        c = weights_forecast_['combo'] == i
        temp = weights_forecast_[c]
        final_pred = f"final_pred_{combo_name[j]}"
        temp = temp[["combo_name", 'loop_back_months_for_weights', 'final_pred', 'forecast_month', 'horizon']].rename(
            columns={"final_pred": final_pred,
                     "combo_name": i
                     })
        print(i)
        print(meta_data_2.dtypes)
        print(temp.dtypes)
        meta_data_2 = pd.merge(meta_data_2, temp, on=[i, "loop_back_months_for_weights", 'forecast_month', 'horizon'],
                               how='left')
        print(meta_data_2.shape)
        print(perc_columns[j])
        print(combo_name[j])
        print(final_pred)
        meta_data_2[f'final_forecast_{combo_name[j]}'] = meta_data_2[perc_columns[j]] * meta_data_2[final_pred]

    ## Attaching combo values for sku forecast
    sku_wise_demand_forecast['loop_back_months_for_weights'] = pd.to_datetime(
        sku_wise_demand_forecast['loop_back_months_for_weights'])
    sku_wise_demand_forecast['forecast_month'] = pd.to_datetime(sku_wise_demand_forecast['forecast_month'])

    on_col = ['loop_back_months_for_weights', 'forecast_month', 'horizon', 'sku', 'loop_number']
    sku_combo_joined = pd.merge(sku_wise_demand_forecast,
                                meta_data_2.drop(['train_data_end', "train_data_start"], axis=1), on=on_col, how='left')

    model_col = ['arima', "simple_exp"] + [f"final_forecast_{'_'.join(i)}" for i in heirarchy_list] + ['actual_demand']

    model_weights = [str(i) + "_weights" for i in model_col[:-1]]
    print(model_col)
    print(model_weights)
    sku_combo_joined.fillna(0, inplace=True)

    for i in model_col:
        #     print(sku_combo_joined[i])
        sku_combo_joined[i] = np.where(np.isinf(sku_combo_joined[i]), 0, sku_combo_joined[i])

    sku_combo_joined.to_csv(output_file_path + "sku_joined_.csv", index=False, sep=',')

    final_prediction = pd.DataFrame()
    for sku_ in tqdm(sku_combo_joined['sku'].unique()):
        sku_condition = sku_combo_joined['sku'] == sku_
        for i in sku_combo_joined['horizon'].unique():
            c1 = sku_combo_joined['horizon'] == i
            temp_a = sku_combo_joined[c1 & sku_condition]
            # clear_output()
            for j in temp_a['forecast_month'].unique():
                c2 = temp_a['forecast_month'] == j
                c3 = temp_a['forecast_month'] != temp_a['loop_back_months_for_weights']
                d, fore_cast_ratio = optimise_weights(temp_a[c2 & c3][model_col], model_col)
                temp_b = temp_a[c2]
                temp_b["final_perdiction"] = temp_b[model_col[:-1]].mul(fore_cast_ratio).sum(axis=1)
                temp_b.reset_index(drop=True, inplace=True)
                ratio_df = pd.DataFrame([fore_cast_ratio] * temp_b.shape[0], columns=model_weights)
                temp_b = pd.concat([temp_b, ratio_df], axis=1)
                #             display(temp_b)
                final_prediction = pd.concat([final_prediction, temp_b], axis=0)

    final_prediction.reset_index(drop=True, inplace=True)

    # CORRECTION FOR FINAL PREDICTION
    final_prediction['final_pred_old'] = final_prediction['final_perdiction']
    final_forecast_ = ["arima", 'simple_exp'] + [f'final_forecast_{"_".join(i)}' for i in heirarchy_list]
    final_forecast_weights = [i + "_weights" for i in final_forecast_]
    # d[final_forecast_+final_forecast_weights]
    final_prediction['final_perdiction'] = (
            pd.DataFrame(final_prediction[final_forecast_].astype(float).values) * pd.DataFrame(
        final_prediction[final_forecast_weights].astype(float).values)).sum(axis=1)

    # # RULES 1 SET USE ARIMA DIRECTLY IN CASE OF MONTHS FROM OCT TO FEB
    # final_prediction['final_perdiction']=np.where(
    #     final_prediction['year_month'].isin(['2023-11','2023-10','2024-01','2023-12','2024-02']),
    #     final_prediction['arima'],final_prediction['final_perdiction'])

    print(final_prediction['year_month'])
    print(final_prediction['forecast_month'])
    final_prediction['year_month']=final_prediction['loop_back_months_for_weights'].astype(str).str[:-3]
    # final_prediction.to_clipboard(index=False,sep=',')
    peak_months = (final_prediction['year_month'].str[-2:]).isin(['10', '11', '12', '01', '02'])
    final_prediction['final_perdiction'] = np.where(
        peak_months,
        final_prediction['arima'], final_prediction['final_perdiction'])

    # IF ARIMA IS ZERO REPLACE IT WITH SIMPLE EXPONENTIAL FORECAST
    final_prediction['arima_is_zero'] = np.where(final_prediction['arima'] == 0, 1, 0)
    final_prediction['final_perdiction'] = np.where(final_prediction['arima'] == 0, final_prediction['simple_exp'],
                                                    final_prediction['final_perdiction'])

    final_prediction = replace_arima_on_peak(final_prediction, peak_data)

    print(output_file_path + 'final_forecast_file.csv',"*"*40)
    final_prediction.to_csv(output_file_path + 'final_forecast_file.csv', index=False, sep=',')
    return final_prediction


def optimize_sku_fore_cast(sku_wise_demand_forecast, output_file_path,peak_data):
    sku_wise_demand_forecast['arima'].fillna(0, inplace=True)
    # second_filter = sku_wise_demand_forecast["sku"].isin(filter_list)
    # sku_wise_demand_forecast = sku_wise_demand_forecast[second_filter]

    sku_combo_joined = sku_wise_demand_forecast.copy()

    model_col = ['arima', "simple_exp"] + ['actual_demand']

    model_weights = [str(i) + "_weights" for i in model_col[:-1]]
    print(model_col)
    print(model_weights)
    sku_combo_joined.fillna(0, inplace=True)

    for i in model_col:
        #     print(sku_combo_joined[i])
        sku_combo_joined[i] = np.where(np.isinf(sku_combo_joined[i]), 0, sku_combo_joined[i])

    final_prediction = pd.DataFrame()
    for sku_ in tqdm(sku_combo_joined['sku'].unique()):
        sku_condition = sku_combo_joined['sku'] == sku_
        for i in sku_combo_joined['horizon'].unique():
            c1 = sku_combo_joined['horizon'] == i
            temp_a = sku_combo_joined[c1 & sku_condition]
            for j in temp_a['forecast_month'].unique():
                c2 = temp_a['forecast_month'] == j
                c3 = temp_a['forecast_month'] != temp_a['loop_back_months_for_weights']
                d, fore_cast_ratio = optimise_weights(temp_a[c2 & c3][model_col], model_col)
                temp_b = temp_a[c2]
                temp_b["final_perdiction"] = temp_b[model_col[:-1]].mul(fore_cast_ratio).sum(axis=1)
                temp_b.reset_index(drop=True, inplace=True)
                ratio_df = pd.DataFrame([fore_cast_ratio] * temp_b.shape[0], columns=model_weights)
                temp_b = pd.concat([temp_b, ratio_df], axis=1)
                #             display(temp_b)
                final_prediction = pd.concat([final_prediction, temp_b], axis=0)

    final_prediction.reset_index(drop=True, inplace=True)

    final_prediction['final_pred_old'] = final_prediction['final_perdiction']
    final_prediction['year_month'] = final_prediction['loop_back_months_for_weights'].astype(str).str[:-3]
    # final_prediction['final_perdiction']=np.where(
    #     final_prediction['year_month'].isin(['2023-11','2023-10','2024-01','2023-12']),
    #     final_prediction['arima'],final_prediction['final_perdiction'])



    peak_months = (final_prediction['year_month'].str[-2:]).isin(['10', '11', '12', '01', '02'])
    final_prediction['final_perdiction'] = np.where(
        peak_months,
        final_prediction['arima'], final_prediction['final_perdiction'])

    # IF ARIMA IS ZERO REPLACE IT WITH SIMPLE EXPONENTIAL FORECAST
    final_prediction['arima_is_zero'] = np.where(final_prediction['arima'] == 0, 1, 0)
    final_prediction['final_perdiction'] = np.where(final_prediction['arima'] == 0, final_prediction['simple_exp'],
                                                    final_prediction['final_perdiction'])

    final_prediction = replace_arima_on_peak(final_prediction, peak_data)

    final_prediction.to_csv(output_file_path + 'final_forecast_file.csv', index=False, sep=',')
    return final_prediction
