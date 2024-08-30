import pandas as pd
import numpy as np
import os

from generate_combo_data import generate_combo_data

from get_product_master_data import get_product_master_data
from utility import make_data_continous

# CAUTION If you get empty data as report please check the current date in the static_variables.py

path = '../FORECAST_CODE/DATA_/BACK_TESTING_NOV_23/'
file_name = 'REPORT_NOV_23'

writer = pd.ExcelWriter(path + file_name + '.xlsx', engine='xlsxwriter')

list_of_categories = os.listdir(path)

continous_input_data=pd.DataFrame()

from static_variables import sales_file_path, max_horizon, current_date

grouped_data = pd.read_csv(sales_file_path)
print(grouped_data.shape)
c = grouped_data['CHNL_NAME'].isin(['Re-Export', 'Export'])
grouped_data['CHNL_NAME'] = np.where(c, 'Export_Re-Export', grouped_data['CHNL_NAME'])
grouped_data = grouped_data.groupby(["MATERIAL", 'year_month', 'CHNL_NAME'], as_index=False)[
    ["QTY_BASEUOM_SUM", "SALES_SUM"]].sum()
print(grouped_data.shape)

sku_master_data = get_product_master_data()
print(sku_master_data.head().T)

grouped_data = pd.merge(grouped_data, sku_master_data[["MATERIAL", "H1"]], on=['MATERIAL'], how='left')

final_forecast_data = pd.DataFrame()

pivot_data = pd.DataFrame()

for cat_folder in os.listdir(path):
    if ('.csv' in cat_folder) | ('.xlsx' in cat_folder):
        continue
    cat = cat_folder
    cat_folder = path + cat_folder + '/'
    for cat_channel_folder in ["B2B", 'Retail', 'Ecomm', 'Export_Re-Export']:
        channel = cat_channel_folder
        print(cat, channel)
        if channel in ['Ecomm', 'Retail']:
            cat_channel_folder = 'Retail_Ecomm'

        if cat_channel_folder == 'Retail_Ecomm':
            forecast_file_name = 'final_forecast_new_ecomm_perc_calculated.csv'
        else:
            forecast_file_name = 'final_forecast_file.csv'

        cat_channel_folder = cat_folder + cat_channel_folder + '/'
        #         print(cat_channel_folder)
        final_path_name = cat_channel_folder + forecast_file_name
        file_exists = os.path.exists(final_path_name)
        print(file_exists, "*" * 40)
        print(final_path_name)
        print(cat, channel)

        cat_filter = grouped_data['H1'] == cat
        channel_filter = grouped_data['CHNL_NAME'] == channel
        input_data = grouped_data[cat_filter & channel_filter].reset_index(drop=True)
        input_data['date'] = pd.to_datetime(input_data['year_month'] + '-01')
        df = make_data_continous(input_data, "MATERIAL", 'QTY_BASEUOM_SUM', max_horizon, current_date)
        df['SALES_SUM'].fillna(0, inplace=True)
        df['CHNL_NAME'] = channel
        df['H1'] = cat
        continous_input_data=pd.concat([continous_input_data,df],axis=0)
        df = pd.merge(df, sku_master_data[['MATERIAL', 'MSTAE']], on=['MATERIAL'], how='left')

        final_prediction_file = pd.read_csv(final_path_name)

        if channel == 'Ecomm':
            final_prediction_file['final_perdiction'] = final_prediction_file['final_perdiction'] * \
                                                        final_prediction_file['ecomm_perc']
        elif channel == "Retail":
            final_prediction_file['final_perdiction'] = final_prediction_file['final_perdiction'] * (
                    1 - final_prediction_file['ecomm_perc'])

        final_prediction_file = pd.merge(final_prediction_file.drop("actual_demand", axis=1),
                                         df.rename(columns={'MATERIAL': "sku"}), on=['sku', 'year_month'], how='left')
        final_prediction_file['CHNL_NAME'] = channel
        final_prediction_file['H1'] = cat
        final_prediction_file['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
        final_prediction_file['SALES_SUM'].fillna(0, inplace=True)
        final_prediction_file.rename(columns={"QTY_BASEUOM_SUM": "actual_demand"}, inplace=True)

        final_prediction_file['forecast_month'] = pd.to_datetime(final_prediction_file['forecast_month'])
        final_prediction_file['loop_back_months_for_weights'] = pd.to_datetime(
            final_prediction_file['loop_back_months_for_weights'])

        c1 = pd.to_datetime(final_prediction_file['train_data_end']) == current_date
        c2 = final_prediction_file['loop_number'] == 0
        # final_prediction_file=
        final_prediction_file = final_prediction_file[c1 & c2]


        final_prediction_file['mape'] = np.abs(
            final_prediction_file['actual_demand'] - final_prediction_file['final_perdiction']) / final_prediction_file[
                                            'actual_demand']
        final_prediction_file.head()

        c1 = final_prediction_file['mape'] <= .25
        c2 = final_prediction_file['mape'] > 0.5
        c3 = (final_prediction_file['mape'] > 0.25) & (final_prediction_file['mape'] <= 0.5)
        final_prediction_file["ACC_BUCKET"] = np.select([c1, c2, c3], ["HIGH", "LOW", 'MEDIUM'], "NA")

        final_prediction_file['ACC_BUCKET'] = np.where(np.isinf(final_prediction_file['mape']), "NA",
                                                       final_prediction_file['ACC_BUCKET'])
        final_prediction_file['mape'] = np.where(np.isinf(final_prediction_file['mape']), "NA",
                                                 final_prediction_file['mape'])

        final_forecast_data = pd.concat([final_forecast_data, final_prediction_file], axis=0)

        pivot_table = pd.DataFrame()
        for i in final_prediction_file['horizon'].unique():
            c = final_prediction_file['horizon'] == i
            c2 = final_prediction_file['loop_number'] == 0

            final_1 = final_prediction_file[c & c2]
            # display(final_1)

            final_1['ACC_BUCKET'].fillna("NA", inplace=True)
            c1 = final_1['ACC_BUCKET'] == 'NA'
            a = pd.pivot_table(final_1, values=['SALES_SUM', 'actual_demand', 'sku'], aggfunc=['sum', 'count'],
                               index=['ACC_BUCKET'])
            b = pd.DataFrame(index=[a.index.values])
            b["actual_demand"] = a['sum']['actual_demand'].values
            b["%actual_demand"] = b['actual_demand'] / b['actual_demand'].sum()
            b['SALES_SUM'] = a['sum']['SALES_SUM'].values
            b['%SALES_SUM'] = b["SALES_SUM"] / b['SALES_SUM'].sum()
            b["sku_count"] = a['count']['sku'].values
            b['sku_count'] = b['sku_count']
            b["%sku_count"] = b['sku_count'] / b['sku_count'].sum()
            f_m = final_1['forecast_month'].unique()[0]
            b['forecast_month'] = f_m.month_name()[:3] + "-" + str(f_m.year)
            b['horizon'] = i
            start_col = ["HIGH", 'MEDIUM', "LOW"]
            start_col = [i for i in start_col if i in b.index]
            #     start_col=[i for i]
            col_left = start_col + [i for i in a.index.values if i not in start_col]
            b = b.loc[col_left, :]
            pivot_table = pd.concat([pivot_table, b], axis=0)

        pivot_table['H1'] = cat
        pivot_table['CHNL_NAME'] = channel

        pivot_data = pd.concat([pivot_data, pivot_table], axis=0)

final_forecast_data.to_excel(writer, sheet_name=file_name,index=False)
pivot_data.to_excel(writer, sheet_name=file_name + "_results")
print(final_forecast_data)
print(pivot_data)
writer.close()
continous_input_data.to_csv(path+"continous_input_file.csv",index=False,sep=',')