import numpy as np
import pandas as pd

from get_product_master_data import get_product_master_data
from static_variables import sales_file_path, current_date, max_horizon
from utility import make_date_continous


def get_ecomm_prec(cat, final_forecast_file,source_path):
    sales = pd.read_csv(sales_file_path)
    sku_detail = get_product_master_data()
    sales['CHNL_NAME'] = np.where(sales['CHNL_NAME'].isin(['Export', 'Re-Export']), "Export-Re-Export",
                                  sales['CHNL_NAME'])

    sales = sales.groupby(['MATERIAL', 'year_month', 'CHNL_NAME'])[['QTY_BASEUOM_SUM', 'SALES_SUM']].sum().reset_index()

    sales = pd.merge(sales, sku_detail, on=['MATERIAL'], how='left')

    # ECOMM
    c = sales['H1'] == cat
    c2 = sales['CHNL_NAME'] == 'Ecomm'
    df = sales[c & c2].reset_index(drop=True)
    ecomm_data = make_date_continous(df, "MATERIAL", 'year_month', current_date)[
        ['MATERIAL', 'year_month', 'QTY_BASEUOM_SUM']]
    ecomm_data['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
    ecomm_data.rename(columns={'QTY_BASEUOM_SUM': "ecomm_sales"}, inplace=True)

    sales_1 = pd.read_csv(sales_file_path)
    sku_detail = get_product_master_data()
    sales_1['CHNL_NEW_NAME'] = np.where(sales_1['CHNL_NAME'].isin(['Retail', 'Ecomm']), "Retail_Ecomm",
                                        sales_1['CHNL_NAME'])
    sales_1 = sales_1.groupby(['MATERIAL', 'year_month', 'CHNL_NEW_NAME'])[
        ['QTY_BASEUOM_SUM', 'SALES_SUM']].sum().reset_index()
    sales_1 = pd.merge(sales_1, sku_detail, on=['MATERIAL'], how='left')

    # ECOMM
    c = sales_1['H1'] == cat
    # c=sales['H1']=='Fruits'
    c2 = sales_1['CHNL_NEW_NAME'] == 'Retail_Ecomm'
    df_1 = sales_1[c & c2].reset_index(drop=True)
    retail_ecomm = make_date_continous(df_1, "MATERIAL", 'year_month', current_date+pd.DateOffset(months=max_horizon))[
        ['MATERIAL', 'year_month', 'QTY_BASEUOM_SUM']]
    retail_ecomm['QTY_BASEUOM_SUM'].fillna(0, inplace=True)
    retail_ecomm.rename(columns={"QTY_BASEUOM_SUM": "sales"}, inplace=True)

    final_input_data = pd.merge(retail_ecomm, ecomm_data, on=['MATERIAL', 'year_month'], how='left')
    final_input_data['ecomm_sales'].fillna(0, inplace=True)

    final_input_data['date'] = pd.to_datetime(final_input_data['year_month'] + '-01')

    final_input_data.sort_values(['MATERIAL', 'date'], inplace=True)

    final_input_data['sku_wise_6_month_sales'] = final_input_data.groupby(['MATERIAL'])['sales'].rolling(7,
                                                                                                         min_periods=1).sum().values
    final_input_data['sku_wise_6_month_sales'] = final_input_data['sku_wise_6_month_sales'] - final_input_data['sales']
    final_input_data['sku_wise_6_month_sales_ecomm'] = final_input_data.groupby(['MATERIAL'])['ecomm_sales'].rolling(7,
                                                                                                                     min_periods=1).sum().values
    final_input_data['sku_wise_6_month_sales_ecomm'] = final_input_data['sku_wise_6_month_sales_ecomm'] - \
                                                       final_input_data['ecomm_sales']
    final_input_data['ecomm_perc'] = final_input_data["sku_wise_6_month_sales_ecomm"] / final_input_data[
        "sku_wise_6_month_sales"]
    final_input_data['ecomm_perc'].fillna(0, inplace=True)

    final_input_data.sort_values(['MATERIAL', 'date'], inplace=True)

    c = final_input_data['date'] > current_date
    final_input_data.loc[c, "ecomm_perc"] = np.nan
    final_input_data['ecomm_perc'] = final_input_data['ecomm_perc'].ffill()

    final_input_data.to_csv(source_path+f"{cat}/Retail_Ecomm/ecomm_retail_input_perc.csv",index=False,sep=',')

    # final_file['loop_number'].unique()
    final_forecast_file['year_month'] = final_forecast_file['loop_back_months_for_weights'].astype(str).str[:-3]

    final_file = pd.merge(final_forecast_file,
                          final_input_data[['MATERIAL', 'year_month', 'ecomm_perc']].rename(
                              columns={'MATERIAL': "sku"}),
                          on=['sku', 'year_month'], how='left'
                          )

    return final_file










