import numpy as np
import pandas as pd
from datetime import datetime

from Croston_Algo import Croston_TSB
from get_clustering_details import get_clustering_details
from get_product_master_data import get_product_master_data
from static_variables import current_date, sales_file_path, filter_a_b_category_file_path
from tqdm import tqdm

from utility import make_date_continous


def get_intermittent_sales_sku_forecast(h1):
    sku_master_data = get_product_master_data()

    grouped_data = pd.read_csv(sales_file_path)
    print(grouped_data.shape)
    exp_rexp = grouped_data['CHNL_NAME'].isin(['Export', 'Re-Export'])
    grouped_data['CHNL_NAME'] = np.where(exp_rexp, "Export_Re-Export", grouped_data['CHNL_NAME'])
    grouped_data = grouped_data.groupby(['MATERIAL', 'year_month', 'CHNL_NAME'], as_index=False)[
        "QTY_BASEUOM_SUM"].sum()
    grouped_data['date'] = grouped_data['year_month'] + '-01'
    grouped_data['date'] = pd.to_datetime(grouped_data['date'])
    c1 = grouped_data['date'] <= current_date
    grouped_data = grouped_data[c1]
    grouped_data = pd.merge(grouped_data, sku_master_data[["MATERIAL", 'H1']], on=['MATERIAL'], how='left')

    d = get_clustering_details(h1, grouped_data)

    d["current_date"] = current_date
    d["current_date"] = pd.to_datetime(d["current_date"])
    d['MaxSellDate'] = pd.to_datetime(d['MaxSellDate'], dayfirst=True)
    d['month_difference'] = (d['current_date'] - d['MaxSellDate']) / np.timedelta64(1, "M")
    d['active_month/age_of_sku'] = (d['active_months'] / d['AGE']).clip(upper=1)
    d['Time since last sale/age'] = d['month_difference'] / d['AGE']
    d['recommended_method'] = np.where(d['AGE'] < 6, "New",
                                       np.where(d['Time since last sale/age'] >= 0.75,
                                                'Croston', np.where(d['active_months'] <= 6, 'Croston', 'SMA')))

    sales_file = pd.read_csv(sales_file_path)
    exp_rexp = sales_file['CHNL_NAME'].isin(['Export', 'Re-Export'])
    sales_file['CHNL_NAME'] = np.where(exp_rexp, "Export_Re-Export", sales_file['CHNL_NAME'])
    sales_file = sales_file.groupby(['MATERIAL', 'year_month', 'CHNL_NAME'], as_index=False)["QTY_BASEUOM_SUM"].sum()
    sales_file = pd.merge(sales_file, sku_master_data[["MATERIAL", "H1"]], on=["MATERIAL"], how='left')

    filter_sku = pd.read_csv(filter_a_b_category_file_path)
    c = filter_sku["Bin - Rev"].isin(['A', "B"])

    sku_list = filter_sku[c]['MATERIAL'].unique()

    data = pd.DataFrame()
    for cat in d['H1'].unique():
        for channel in d['CHNL_NAME'].unique():
            print(cat, channel)
            c1 = sales_file['CHNL_NAME'] == channel
            c2 = sales_file['H1'] == cat
            c3 = sales_file["MATERIAL"].isin(sku_list) == False
            df = sales_file[c1 & c2 & c3]
            print(df.shape)
            print(df['MATERIAL'].nunique())
            df = df[['MATERIAL', 'year_month', 'QTY_BASEUOM_SUM', ]]
            df = make_date_continous(df, 'MATERIAL', 'year_month', datetime(2024, 5, 1))
            df['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
            col1 = ['horizon', 'sku', 'train_data_end', 'train_data_start', 'forecast_month', 'final_perdiction',
                    'actual_demand', 'SALES_SUM', 'mape', 'mape_overall', 'ACC_BUCKET', 'arima', 'simple_exp',
                    'Holt_linear', 'Holt_additive_damped', 'croston', 'arima_weights', 'simple_exp_weights',
                    'Simple_Moving_Average', 'Final Prediction']
            dataframes = pd.DataFrame(columns=col1)

            channels_df = d[d['CHNL_NAME'] == channel]

            for sku in tqdm(df["MATERIAL"].unique()):
                df_final = pd.DataFrame(columns=col1)
                df_final['horizon'] = list(range(1, 7, 1))
                df_final['sku'] = sku
                next_months = list((pd.date_range(current_date, current_date + pd.DateOffset(months=6),
                                                  freq="M") + pd.DateOffset(days=1)).astype(str).str[:-3])
                df_final['forecast_month'] = next_months

                # Simple Moving Average
                daf = df[df['MATERIAL'] == sku]
                daf_croston = daf.copy()
                year_months = list((pd.date_range(current_date - pd.DateOffset(months=6), current_date,
                                                  freq="M") + pd.DateOffset(days=1)).astype(str).str[:-3])
                daf = daf[daf['year_month'].isin(year_months)]

                avg_sales = daf[daf['year_month'].isin(year_months)]['QTY_BASEUOM_SUM'].mean()

                df_final['Simple_Moving_Average'] = list(np.repeat(avg_sales, 6))

                # Croston_TSB
                # print(sku)
                # print(channel)
                # print(daf['QTY_BASEUOM_SUM'], 'Preinty daf')
                daf_croston['date'] = pd.to_datetime(daf_croston['year_month'] + '-01')
                c = daf_croston['date'] <= current_date
                daf_croston = daf_croston[c]
                if daf_croston.shape[0] == 0:
                    continue

                cdf = Croston_TSB(daf_croston['QTY_BASEUOM_SUM'], extra_periods=6, alpha=0.5, beta=0.7)
                cros = list(cdf['Forecast'].values[-6:])

                df_final['croston'] = cros

                match_df = channels_df[channels_df['sku'] == sku]

                if match_df.empty:
                    print(f"Warning: No matching SKU found in channels_df for SKU: {sku}")
                    continue

                if not match_df.empty:
                    if match_df['recommended_method'].values[0] == 'SMA':
                        df_final['Final Prediction'] = list(np.repeat(avg_sales, 6))
                    elif match_df['recommended_method'].values[0] == 'Croston':
                        df_final['Final Prediction'] = cros
                #             clear_output()
                df_final['Channel'] = channel
                df_final['H1'] = cat

                df_final['recommended_method'] = match_df['recommended_method'].values[0]
                dataframes = pd.concat([dataframes, df_final], ignore_index=True)
            data = pd.concat([data, dataframes], axis=0)
    # data.to_clipboard(index=False, sep=',')
    return data

# get_intermittent_sales_sku_forecast("Dairy")
