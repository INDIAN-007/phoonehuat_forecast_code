import numpy as np
import pandas as pd

from static_variables import current_date
from utility import make_date_continous, clustering_coefficient


def get_clustering_details(cat, data):
    # cat = "Dairy"
    c = data['H1'] == cat
    df = data[c]
    df.head()

    data_ = pd.DataFrame()
    for i in df['CHNL_NAME'].unique():
        print(i)
        df_dict = {}
        c = df['CHNL_NAME'] == i
        dd = df[c]
        dd.drop(['H1', 'CHNL_NAME', 'date'], axis=1, inplace=True)
        #     display(dd.head())
        dd.rename(columns={"QTY_BASEUOM_SUM": "sales", "MATERIAL": "sku"}, inplace=True)
        dd['date_'] = pd.to_datetime(dd['year_month'] + '-01')
        dd['active_month'] = (dd['sales'] > 0).astype(int)
        c = dd['date_'] <= current_date
        dd = dd[c]
        dd_ = dd.groupby(['sku']).agg({'date_': ['min', 'max'], 'active_month': ['sum']}).reset_index()
        dd_.columns = ['sku', 'min_date', 'max_date', 'active_months']
        # print((current_date - dd_['min_date']))
        # print(pd.to_timedelta(current_date - dd_['min_date']))
        # print((current_date - dd_['min_date'])/ np.timedelta64(1, "M"))
        dd_["AGE"] = np.ceil((current_date - dd_['min_date']) / np.timedelta64(1, "M")) + 1

        dd_c = make_date_continous(dd, 'sku', 'year_month', current_date)
        dd_c['sales'].fillna(0, inplace=True)
        #     dd_c['SALES_SUM'].fillna(0,inplace=True)
        dd_c['active_month'].fillna(0, inplace=True)

        dict_ = {
            "sku": [],
            'clustering_metric': []
        }
        for j in dd_c['sku'].unique():
            c = dd_c['sku'] == j
            cc = clustering_coefficient(dd_c[c]['active_month'])
            dict_['sku'].append(j)
            dict_['clustering_metric'].append(cc)

        dd_ = pd.merge(dd_, pd.DataFrame(dict_), on='sku', how='left')
        dd_['CHNL_NAME'] = i
        data_ = pd.concat([data_, dd_], axis=0)

    data_.rename(columns={'max_date': "MaxSellDate"}, inplace=True)
    data_['H1'] = cat
    return data_
