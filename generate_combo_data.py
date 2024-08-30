import pandas as pd

from utility import make_data_continous


def generate_combo_data(df, heirarchy_list, current_date):
    df_ = df.copy()
    df_['date'] = pd.to_datetime(df_['year_month'] + "-01")
    df_.sort_values(['sku', 'date'], inplace=True)
    df_['6_months_sku_sales'] = df_.groupby('sku')['sales'].rolling(6).sum().values
    # df_['6_months_sku_sales']=df_['6_months_sku_sales']-df_['sales']
    combo_data = pd.DataFrame()
    for i in heirarchy_list:
        print(heirarchy_list)
        cols = '_'.join(i)
        print(cols)
        combo_column_name = f'combo_name_{cols}'
        sales_combo_column_name = f'sales_{cols}'
        print(combo_column_name, sales_combo_column_name)
        #    Gather the COMBOS
        df_[combo_column_name] = df_[i].astype(str).sum(axis=1)
        combo_groupby_data = df_.groupby([combo_column_name, 'year_month'])['sales'].sum().reset_index()
        combo_groupby_data['date'] = pd.to_datetime(combo_groupby_data['year_month'] + "-01")

        combo_groupby_data_continous = make_data_continous(combo_groupby_data, combo_column_name, 'sales', 6,
                                                           current_date)
        combo_groupby_data_continous.drop('date', axis=1, inplace=True)
        combo_groupby_data_continous[f'6_month_sales_{cols}'] = combo_groupby_data_continous.groupby(combo_column_name)[
            "sales"].rolling(6).sum().values
        combo_groupby_data_continous[f'6_month_sales_{cols}'] = combo_groupby_data_continous[f'6_month_sales_{cols}']
        # -combo_groupby_data_continous['sales']
        combo_groupby_data_continous[f'6_month_sales_{cols}'].fillna(0, inplace=True)
        combo_groupby_data_continous.rename(columns={"sales": sales_combo_column_name}, inplace=True)
        df_ = pd.merge(df_, combo_groupby_data_continous, on=[combo_column_name, 'year_month'], how='left')
        df_[f'perc_sales_{cols}'] = df_['6_months_sku_sales'] / df_[f'6_month_sales_{cols}']
        df_[f'perc_sales_{cols}'].fillna(0, inplace=True)
    return df_