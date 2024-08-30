import pandas as pd
import numpy as np
from tqdm import tqdm

from get_product_master_data import get_product_master_data
from static_variables import sales_file_path, max_horizon
from utility import make_data_continous
import statsmodels.api as sm
from static_variables import current_date


def generate_peak_months_data():
    input_data = pd.read_csv(sales_file_path)
    sku_master = get_product_master_data()
    print(sku_master.head())

    cc = input_data['CHNL_NAME'].isin(['Export', 'Re-Export'])
    input_data['CHNL_NAME'] = np.where(cc, "Export_Re-Export", input_data['CHNL_NAME'])
    input_data = input_data.groupby(['MATERIAL', 'CHNL_NAME', 'year_month'], as_index=False).sum()

    cc = input_data['CHNL_NAME'].isin(['Retail', 'Ecomm'])
    input_data['CHNL_NAME'] = np.where(cc, "Retail_Ecomm", input_data['CHNL_NAME'])
    input_data = input_data.groupby(['MATERIAL', 'CHNL_NAME', 'year_month'], as_index=False).sum()

    input_data = pd.merge(input_data, sku_master[['MATERIAL', "H1"]], on=['MATERIAL'], how='left')
    print(input_data.head())
    data_peaks = pd.DataFrame()
    for cat_ in tqdm(input_data['H1'].unique()):

        for channel_ in input_data['CHNL_NAME'].unique():

            dict_ = {"sku": [], 'months_peak': [], 'months_low': []}

            cat = input_data['H1'] == cat_
            channel = input_data['CHNL_NAME'] == channel_
            h1_level_data = input_data[cat & channel].reset_index(drop=True)
            if h1_level_data.shape[0] == 0:
                continue
            df = make_data_continous(h1_level_data, 'MATERIAL', 'QTY_BASEUOM_SUM', max_horizon,
                                     current_date - pd.DateOffset(months=max_horizon))
            print(df.shape)
            print(df.head(), 'printing')
            date_ = pd.to_datetime(df['year_month'] + '-01')
            c = date_ <= current_date
            df = df[c]

            df['QTY_BASEUOM_SUM'].fillna(0, inplace=True)

            for sku in df['MATERIAL'].unique():
                c = df['MATERIAL'] == sku
                temp = df[c]
                try:
                    decompose = sm.tsa.seasonal_decompose(temp['QTY_BASEUOM_SUM'], period=12)
                    temp['seasonal'] = np.abs(decompose.seasonal)
                    temp['seasonal_na'] = decompose.seasonal
                    temp['SI'] = temp['seasonal'] / temp['seasonal'].mean()
                    temp['SI_NA'] = temp['seasonal_na'] / temp['seasonal'].mean()
                    c = temp['SI_NA'] > 1
                    c1 = temp['SI_NA'] < -1
                    # months_=list(temp[c]["months"].unique())
                    # dict_['sku'].append(sku)
                    temp['months'] = temp['year_month'].str[5:]
                    temp['months'] = temp['months'].astype(int)
                    for m in temp[c]["months"].unique():
                        dict_['sku'].append(sku)
                        dict_['months_peak'].append(m)
                        dict_['months_low'].append(np.nan)

                    for m in temp[c1]["months"].unique():
                        dict_['sku'].append(sku)
                        dict_['months_low'].append(m)
                        dict_['months_peak'].append(np.nan)

                except:
                    print(temp.shape)
                    dict_['sku'].append(sku)
                    dict_['months_low'].append(f'age of sku is {temp.shape[0]}')
                    dict_['months_peak'].append(f'age of sku is {temp.shape[0]}')
            print(cat_, channel_)
            # print(len(dict_['sku']),len(dict_['months']))
            peak_data = pd.DataFrame(dict_)
            peak_data["H1"] = cat_
            peak_data["Channel"] = channel_
            data_peaks = pd.concat([data_peaks, peak_data], axis=0)

    return data_peaks


generate_peak_months_data()
