# SKU PIPIELINE 1 EXP REXP
import os
import warnings

from generate_combo_data import generate_combo_data
from generate_combo_level_forecast import generate_combo_level_forecast
from generate_combo_meta_data import generate_combo_meta_data
from generate_meta_data import genearate_meta_data_new
from get_sku_level_forecast import get_sku_level_forecast
from static_variables import product_hierarchy_path, max_horizon, loop_back_time, current_date, \
    main_path, \
    sales_file_path, production, heirarchy_list, categories_for_combo
from utility import collect_data, create_directory_if_not_present, make_data_continous

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from tqdm import tqdm

sales_file = pd.read_csv(sales_file_path,
                         usecols=["MATERIAL", "year_month", "CHNL_NAME", "QTY_BASEUOM_SUM"],
                         dtype={"MATERIAL": np.int32, "QTY_BASEUOM_SUM": np.int32}

                         )

import os

product_hierarchy_files = os.listdir(product_hierarchy_path)
col = ['Product', 'H1_description', "H2_description", "H3_description", "H4_description",
       "H5_description", 'PdProductDescription', 'CrossPlantStatus', 'NetWeight']
sku_master_data = collect_data(product_hierarchy_path, product_hierarchy_files, col=col)
sku_master_data.rename(columns={'Product': "MATERIAL", 'H1_description': "H1", "H2_description": "H2"
    , "H3_description": 'H3', "H4_description": "H4", "H5_description": "H5",
                                'NetWeight': "KG", "CrossPlantStatus": "MSTAE"}, inplace=True)

sku_master_data['MATERIAL'] = sku_master_data['MATERIAL'].astype(int)

sales_file = pd.merge(sales_file, sku_master_data[["MATERIAL", "H1"]], on=["MATERIAL"], how='left')


# def get_forecast_for_category(input_file, h1, channel, path):
#     meta_data = genearate_meta_data_new(input_file, max_horizon, loop_back_time, current_date)
#
#     if production:
#         c = meta_data['loop_number'] == 0
#         meta_data = meta_data[c]
#     meta_data.to_csv(path + f'meta_data.csv', index=False, sep=',')
#     sku_wise_demand_forecast = pd.DataFrame()
#     input_file['date_'] = input_file['date']
#
#     for i in tqdm(input_file['sku'].unique()):
#         a = get_sku_level_forecast(i, current_date, input_file, meta_data, max_horizon)
#         sku_wise_demand_forecast = pd.concat([sku_wise_demand_forecast, a], axis=0)
#         print(sku_wise_demand_forecast.shape, "Prnting shape of sku-wise")
#     sku_wise_demand_forecast.to_csv(path + f'sku_wise_demand_forecast.csv', index=False, sep=',')


# FILTER SECTION
filter_df = pd.read_csv('../DATA_MILAN/filter_BIN_.csv')
a_b_category_filter_condition = filter_df['Bin - Rev'].isin(['A', "B"])
sku_list = filter_df[a_b_category_filter_condition]['MATERIAL'].unique()

# filter_bin = pd.read_csv("./DATA_MILAN/filter_BIN_.csv")
# c1 = filter_bin['Bin - Rev'].isin(['A', 'B'])

for cat_ in categories_for_combo:
    for chnl in categories_for_combo[cat_]:
        channel_list = chnl.split("_")
        print(cat_, channel_list)
        create_directory_if_not_present(main_path + f'DATA_/H1/{cat_}/{chnl}')
        source_path = main_path + f'DATA_/H1/{cat_}/{chnl}/'
        print(source_path, 'Print')
        category_filter = sales_file['H1'] == cat_
        channel_filter = sales_file['CHNL_NAME'].isin(channel_list)
        input_file = sales_file[category_filter & channel_filter]
        if input_file.shape[0] == 0:
            print(input_file.shape)
            continue

        input_file = input_file.groupby(["MATERIAL", 'year_month'], as_index=False)['QTY_BASEUOM_SUM'].sum()
        #         input_file['date']=pd.to_datetime(input_file['year_month']+'-01')
        input_file = make_data_continous(input_file, "MATERIAL", 'QTY_BASEUOM_SUM', 6, current_date)
        input_file['Channel'] = chnl
        #         input_file['H1']=cat_
        input_file.reset_index(drop=True, inplace=True)

        input_file.rename(columns={"MATERIAL": "sku", 'QTY_BASEUOM_SUM': "sales"}, inplace=True)
        input_file = pd.merge(input_file, sku_master_data, left_on=['sku'],right_on=['MATERIAL'], how='left')

        number = input_file['KG'].astype(float).astype(int).astype(str)

        decimal_ = input_file['KG'].astype(str).str.split('.').str[1].str.replace(r'(?<!^0)0+$', '',
                                                                                  regex=True).replace("", "0")

        input_file['KG'] = number + "." + decimal_

        input_file.drop('date', axis=1, inplace=True)

        combo_data = generate_combo_data(input_file, heirarchy_list, current_date)

        combo_data.to_csv(source_path + f'combo_data.csv', index=False, sep=',')

        combo_meta_data = generate_combo_meta_data(combo_data, heirarchy_list, current_date, max_horizon,
                                                   loop_back_time)

        combo_meta_data.to_csv(source_path + f'combo_meta_data.csv', index=False, sep=',')

        if production:
            c = combo_meta_data['loop_number'] == 0
            combo_meta_data = combo_meta_data[c]

        c2 = combo_data['sku'].isin(sku_list)
        combo_data = combo_data[c2]

        combo_level_forecast = generate_combo_level_forecast(heirarchy_list, combo_data, combo_meta_data, max_horizon,
                                                             current_date)
        combo_level_forecast.to_csv(source_path + f'combo_wise_demand_forecast.csv', index=False, sep=',')