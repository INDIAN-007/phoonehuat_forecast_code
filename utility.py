import pandas as pd
import numpy as np

from static_variables import max_horizon


def collect_data(path, files, col=None):
    d = pd.DataFrame()
    for i in files:
        file_path = path + i
        print(file_path)
        data = pd.read_parquet(file_path, columns=col)
        print(data.shape)
        d = pd.concat([d, data], axis=0)
    return d.reset_index(drop=True)


def make_data_continous(data_, groupby_col, sales_col, max_horizon, current_date):
    data_['date'] = pd.to_datetime(data_['year_month'] + '-01')
    continous_data = pd.DataFrame()
    for sku_no in data_[groupby_col].unique():
        # print(sku_no)
        sku_filter = data_[groupby_col] == sku_no
        data = data_[sku_filter]
        min_date = data['date'].min() - pd.DateOffset(months=1)
        max_date = current_date + pd.DateOffset(months=max_horizon)
        date_range = pd.date_range(min_date, max_date, freq='M') + pd.DateOffset(days=1)
        temp_data = pd.DataFrame()
        temp_data['date'] = date_range
        data = pd.merge(temp_data, data, on=['date'], how='left')
        data[groupby_col] = sku_no
        continous_data = pd.concat([continous_data, data], axis=0)
    continous_data['year_month'] = continous_data['date'].astype(str).str[:-3]
    continous_data[sales_col].fillna(0, inplace=True)

    return continous_data


def create_directory_if_not_present(path):
    import os
    # print(path)
    path_list = path.split("/")
    # print(path_list)
    path_ = ""
    for i in path_list:
        path_ += i + '/'
        print(path_)

        path_exists = os.path.exists(path_)
        if not path_exists:
            os.mkdir(path_)
        # print(path_exists)


def make_date_continous(data, level_col, year_month_col, current_date):
    main_data = pd.DataFrame()
    data['date'] = pd.to_datetime(data[year_month_col] + "-01")
    for i in data[level_col].unique():
        c = data[level_col] == i
        temp = data[c]
        min_date = temp['date'].min()
        date_range = pd.date_range(min_date - pd.DateOffset(months=1), current_date, freq='M') + pd.DateOffset(days=1)
        continous_dates = pd.DataFrame()
        continous_dates['date'] = date_range
        continous_dates = pd.merge(continous_dates, temp, on='date', how='left')
        continous_dates[level_col] = i
        main_data = pd.concat([main_data, continous_dates], axis=0)
    main_data['year_month'] = main_data['date'].astype(str).str[:-3]
    del main_data['date']
    return main_data


def clustering_coefficient(sequence):
    #     print(sequence)
    n = len(sequence)
    if n < 2:
        return 0
    total_ones = sum(sequence)
    if total_ones == 0:
        return 0
    adjacent_ones = 0
    for i in range(1, n):
        if sequence[i] == 1 and sequence[i - 1] == 1:
            adjacent_ones += 1
    # The maximum number of adjacent pairs of 1s is total_ones - 1 (in a perfect cluster)
    max_adjacent_ones = total_ones - 1
    if max_adjacent_ones == 0:
        return 0
    clustering_coeff = adjacent_ones / max_adjacent_ones
    return clustering_coeff


def fill_horizon_pred(temp_meta_data_, i, model_name, values):
    for horizon in range(1, max_horizon + 1):
        c1 = temp_meta_data_['horizon'] == horizon
        c2 = temp_meta_data_['train_data_end'] == i
        s = temp_meta_data_[c1 & c2]
        if s.shape[0] != 0:
            temp_meta_data_.loc[c1 & c2, model_name] = values[horizon - 1] if values[horizon - 1] else 0


#             croston if croston > 0 else 0


def replace_arima_on_peak(final_forecast, peak_data):
    final_forecast['months'] = final_forecast['year_month'].str[-2:].astype(int)
    final_forecast = pd.merge(final_forecast, peak_data[['sku', 'months', 'stats', 'months_low', 'months_peak']],
                              on=['sku', 'months'], how='left')
    final_forecast['stats'].fillna(0, inplace=True)
    c = final_forecast['arima_is_zero'] != 1
    c2 = final_forecast['stats'].astype(float).astype(int) == 1
    final_forecast["final_perdiction"] = np.where(c & c2, final_forecast['arima'], final_forecast['final_perdiction'])
    return final_forecast
