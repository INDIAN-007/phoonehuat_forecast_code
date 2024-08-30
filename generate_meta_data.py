def genearate_meta_data_new(df_,max_horizon,loop_back_time,current_date):
    import pandas as pd
    df=df_.copy()
    df['date__']=pd.to_datetime(df['year_month']+"-01")
    dict_={
        "loop_back_months_for_weights":[],
         "loop_number":[],
        'train_data_end':[],
        'train_data_start':[],
        'forecast_month':[],
        'horizon':[],
        'sku':[]
    #     'forecast_month':[]
    }
    for sku_ in df['sku'].unique():
        c=df['sku']==sku_
        temp=df[c]
    #     display(temp)
        min_date=temp['date__'].min()

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
                    dict_['sku'].append(sku_)
                    dict_['loop_number'].append(j)


        #         print(loop_back_month_forecast_month+pd.DateOffset(months=horizon),loop_back_month_forecast_month)
                dict_['loop_back_months_for_weights'].append(loop_back_month_forecast_month+pd.DateOffset(months=horizon))
                dict_['train_data_end'].append(loop_back_month_forecast_month)
                dict_['train_data_start'].append(min_date)
                dict_['forecast_month'].append(loop_back_month_forecast_month+pd.DateOffset(months=horizon))
                dict_['horizon'].append(horizon)
                dict_['sku'].append(sku_)
                dict_['loop_number'].append(j)
    #         print()

    return pd.DataFrame(dict_)

