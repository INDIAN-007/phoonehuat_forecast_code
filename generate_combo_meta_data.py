# New Corrected Combo Meta data concept
import pandas as pd
import numpy as np
def generate_combo_meta_data(combo_data_,heirarchy_list,current_date,max_horizon,loop_back_time):
    combo_data_['date']=pd.to_datetime(combo_data_['year_month']+'-01')
    dict_={
        "loop_back_months_for_weights":[],
        "train_data_end":[],
        "train_data_start":[],
        "forecast_month":[],
        "horizon":[],
        "loop_number":[],
        "combo_name":[],
        "combo":[]
    }
    combo_columns_list=[f"combo_name_{'_'.join(i)}" for i in heirarchy_list]
    for combination_column in combo_columns_list:
    #     print(combination_column)
        for combo_name_ in combo_data_[combination_column].unique():
    #         print(combo_name_)
            c=combo_data_[combination_column]==combo_name_
            min_date=combo_data_[c]['date'].min()
            min_date=min_date.date()
            for horizon in range(1,max_horizon+1):
                for j in range(loop_back_time):
                    loop_back_month_forecast_month=current_date-pd.DateOffset(months=j)
                    for i in range(loop_back_time):
                        loop_back_month=loop_back_month_forecast_month-pd.DateOffset(months=i)
            #             print(loop_back_month,loop_back_month-pd.DateOffset(months=horizon))
                        dict_['loop_back_months_for_weights'].append(loop_back_month)
                        dict_['train_data_end'].append(loop_back_month-pd.DateOffset(months=horizon))
                        dict_['train_data_start'].append(min_date)
                        dict_['forecast_month'].append(loop_back_month_forecast_month+pd.DateOffset(months=horizon))
                        dict_['horizon'].append(horizon)
                        dict_['combo_name'].append(combo_name_)
                        dict_['combo'].append(combination_column)
                        dict_['loop_number'].append(j)


            #         print(loop_back_month_forecast_month+pd.DateOffset(months=horizon),loop_back_month_forecast_month)
                    dict_['loop_back_months_for_weights'].append(loop_back_month_forecast_month+pd.DateOffset(months=horizon))
                    dict_['train_data_end'].append(loop_back_month_forecast_month)
                    dict_['train_data_start'].append(min_date)
                    dict_['forecast_month'].append(loop_back_month_forecast_month+pd.DateOffset(months=horizon))
                    dict_['horizon'].append(horizon)
                    dict_['combo_name'].append(combo_name_)
                    dict_['combo'].append(combination_column)
                    dict_['loop_number'].append(j)
    return pd.DataFrame(dict_)
