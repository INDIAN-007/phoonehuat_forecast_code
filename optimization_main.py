import numpy as np
import pandas as pd

from calculate_retail_ecomm_prec import get_ecomm_prec
from generate_combo_data import generate_combo_data
from generate_combo_meta_data import generate_combo_meta_data
from generate_meta_data import genearate_meta_data_new
from generate_peak_months_data import generate_peak_months_data
from get_product_master_data import get_product_master_data
from intermittent_sales_sku_forecast import get_intermittent_sales_sku_forecast
from optimization_sku_combo_forecast import optimize_sku_combo_forecast, optimize_sku_fore_cast
from static_variables import categories_for_sku, categories_for_combo, main_path, heirarchy_list, current_date, \
    sales_file_path, max_horizon, loop_back_time
from utility import make_data_continous, replace_arima_on_peak

# './DATA_/H2/Dairy/Retail_Eco'
# FILTER SECTION
filter_df = pd.read_csv('./INPUT_FILES/filter_BIN_.csv')
a_b_category_filter_condition = filter_df['Bin - Rev'].isin(['A', "B"])
sku_list = filter_df[a_b_category_filter_condition]['MATERIAL'].unique()

sales_file = pd.read_csv(sales_file_path,
                         usecols=["MATERIAL", "year_month", "CHNL_NAME", "QTY_BASEUOM_SUM"],
                         dtype={"MATERIAL": np.int32, "QTY_BASEUOM_SUM": np.int32}

                         )

# GETTING DATA PEAK MONTHS DATA -START
peak_data = generate_peak_months_data()
peak_data.reset_index(drop=True, inplace=True)
peak_data.to_csv('./DATA_/data_peak.csv', index=False)

# peak_data=pd.read_csv('./DATA_/data_peak.csv')


c = peak_data['months_low'].isna() | peak_data['months_peak'].isna()
data_peak_1 = peak_data[c]

data_peak_1['months_low'].fillna(0, inplace=True)
data_peak_1['months_peak'].fillna(0, inplace=True)
data_peak_1['months'] = data_peak_1['months_low'] + data_peak_1['months_peak']
data_peak_1['stats'] = 1
data_peak_1['months'] = data_peak_1['months'].astype(int)

# GETTING DATA PEAK MONTHS -END


sku_master_data = get_product_master_data()
sku_master_data['MATERIAL'] = sku_master_data['MATERIAL'].astype(int)

sales_file = pd.merge(sales_file, sku_master_data[["MATERIAL", "H1"]], on=["MATERIAL"], how='left')
# PATH TO CHANGE
source_path = main_path + 'DATA_/NOV_23_MAY_24_SEASONALITY/'
print(source_path)
collective_final_forecast = pd.DataFrame()
col_to_be_present = ['train_data_end',
                     'train_data_start', 'horizon', 'sku',
                     'year_month', 'final_perdiction', 'H1', "Channel"]
for cat_ in categories_for_sku:
    for chnl in categories_for_sku[cat_]:
        # continuez
        channel_list = chnl.split("_")
        print(cat_, chnl)
        a = categories_for_combo[cat_]
        combo_approach = chnl in a
        category_filter = sales_file['H1'] == cat_
        channel_filter = sales_file['CHNL_NAME'].isin(channel_list)
        input_file = sales_file[category_filter & channel_filter]
        if input_file.shape[0] == 0:
            # print(input_file.shape)
            continue

        input_file = input_file.groupby(["MATERIAL", 'year_month'], as_index=False)['QTY_BASEUOM_SUM'].sum()
        #         input_file['date']=pd.to_datetime(input_file['year_month']+'-01')
        input_file = make_data_continous(input_file,
                                         "MATERIAL",
                                         "QTY_BASEUOM_SUM",
                                         6,
                                         current_date)
        input_file['Channel'] = chnl
        input_file['H1'] = cat_
        input_file.reset_index(drop=True, inplace=True)
        input_file_filter_condition = input_file['MATERIAL'].isin(sku_list)
        meta_data = genearate_meta_data_new(input_file.rename(columns={"MATERIAL": "sku"}), max_horizon, loop_back_time,
                                            current_date)

        sku_wise_demand_forecast = pd.read_csv(source_path + f"{cat_}/{chnl}/sku_wise_demand_forecast.csv")
        # meta_data = pd.read_csv(source_path + f"{cat_}/{chnl}/meta_data.csv")
        if combo_approach:
            input_file.rename(columns={"MATERIAL": "sku", 'QTY_BASEUOM_SUM': "sales"}, inplace=True)
            input_file = pd.merge(input_file, sku_master_data, left_on=['sku'], right_on=['MATERIAL'], how='left')

            number = input_file['KG'].astype(float).astype(int).astype(str)

            decimal_ = input_file['KG'].astype(str).str.split('.').str[1].str.replace(r'(?<!^0)0+$', '',
                                                                                      regex=True).replace("", "0")

            input_file['KG'] = number + "." + decimal_

            input_file.drop('date', axis=1, inplace=True)

            combo_data = generate_combo_data(input_file, heirarchy_list, current_date)

            combo_meta_data = generate_combo_meta_data(combo_data, heirarchy_list, current_date, max_horizon,
                                                       loop_back_time)

            # combo_data = pd.read_csv(source_path + f"{cat_}/{chnl}/combo_data.csv")
            # combo_meta_data = pd.read_csv(source_path + f"{cat_}/{chnl}/combo_meta_data.csv")

            combo_wise_demand_forecast = pd.read_csv(source_path + f"{cat_}/{chnl}/combo_wise_demand_forecast.csv")
            perc_sales = ['perc_sales_' + "_".join(i) for i in heirarchy_list]
            combo_data['date'] = pd.to_datetime(combo_data['year_month'] + '-01')
            c2 = combo_data['date'] > current_date
            combo_data.loc[c2, perc_sales] = np.nan
            for i in perc_sales:
                combo_data[i] = combo_data[i].ffill()
            final_forecast = optimize_sku_combo_forecast(combo_wise_demand_forecast, meta_data,
                                                         combo_data, sku_wise_demand_forecast,
                                                         source_path + f"{cat_}/{chnl}/",
                                                         data_peak_1[data_peak_1['Channel'] == chnl])
            # Replacing peaks with ARIMA
            # final_forecast = replace_arima_on_peak(final_forecast, data_peak_1[data_peak_1['Channel'] == chnl])

            if chnl == "Retail_Ecomm":
                final_forecast_1 = get_ecomm_prec(cat_, final_forecast, source_path)
                final_forecast_1.to_csv(source_path + f"{cat_}/{chnl}/final_forecast_new_ecomm_perc_calculated.csv",
                                        index=False, sep=',')
                retail_forecast = final_forecast_1.copy()

                c = retail_forecast['loop_back_months_for_weights'] == retail_forecast['forecast_month']
                retail_forecast = retail_forecast[c]
                retail_forecast['final_perdiction'] = (1 - retail_forecast['ecomm_perc']) * retail_forecast[
                    'final_perdiction']
                retail_forecast['H1'] = cat_
                retail_forecast['Channel'] = 'Retail'
                collective_final_forecast = pd.concat([collective_final_forecast,
                                                       retail_forecast[col_to_be_present]], axis=0)
                ecomm_forecast = final_forecast_1.copy()
                c = ecomm_forecast['loop_back_months_for_weights'] == ecomm_forecast['forecast_month']
                ecomm_forecast = ecomm_forecast[c]
                ecomm_forecast['final_perdiction'] = ecomm_forecast['ecomm_perc'] * ecomm_forecast[
                    'final_perdiction']
                ecomm_forecast['H1'] = cat_
                ecomm_forecast['Channel'] = "Ecomm"

                collective_final_forecast = pd.concat([collective_final_forecast,
                                                       ecomm_forecast[col_to_be_present]], axis=0)

            else:
                final_forecast['H1'] = cat_
                final_forecast['Channel'] = chnl
                c = final_forecast['loop_back_months_for_weights'] == final_forecast['forecast_month']
                final_forecast = final_forecast[c]

                collective_final_forecast = pd.concat([collective_final_forecast,
                                                       final_forecast[col_to_be_present]],
                                                      axis=0)
        else:
            final_forecast = optimize_sku_fore_cast(sku_wise_demand_forecast, source_path + f"{cat_}/{chnl}/"
                                                    , data_peak_1[data_peak_1['Channel'] == chnl])
            final_forecast['H1'] = cat_
            final_forecast['Channel'] = chnl
            c = final_forecast['loop_back_months_for_weights'] == final_forecast['forecast_month']
            final_forecast = final_forecast[c]

            # final_forecast = replace_arima_on_peak(final_forecast, data_peak_1[data_peak_1['Channel'] == chnl])

            collective_final_forecast = pd.concat([collective_final_forecast,
                                                   final_forecast[col_to_be_present]],
                                                  axis=0)
    print(len(cat_))

    croston_files = get_intermittent_sales_sku_forecast(cat_)
    # croston_files = pd.DataFrame(columns=['final_perdiction', 'Final Prediction', 'forecast_month'])
    # col_to_be_present=['final_perdiction', 'forecast_month']
    print(croston_files.shape)

    croston_files.drop('final_perdiction', axis=1, inplace=True)
    croston_files.rename(columns={"Final Prediction": "final_perdiction"}, inplace=True)
    croston_files['year_month'] = croston_files['forecast_month']
    croston_files['forecast_month'] = pd.to_datetime(croston_files['forecast_month'] + "-01")
    c = croston_files['final_perdiction'].notnull()
    croston_files = croston_files[c]
    croston_files = croston_files[col_to_be_present]

    collective_final_forecast = pd.concat([collective_final_forecast,
                                           croston_files],
                                          axis=0)

collective_final_forecast.to_csv(source_path + 'combined_file_new.csv'
                                 , index=False, sep=',')
