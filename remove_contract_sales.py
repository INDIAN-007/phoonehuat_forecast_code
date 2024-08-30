import os
import pandas as pd
import numpy as np
master_path='./INPUT_FILES/product_master_files/'
master_files=os.listdir(master_path)
def collect_data(path,files,col=None):
    d=pd.DataFrame()
    for i in files:
        file_path=path+i
        print(file_path)
        data=pd.read_parquet(file_path,columns=col)
        print(data.shape)
        d=pd.concat([d,data],axis=0)
    return d.reset_index(drop=True)

master=collect_data(master_path,master_files)

rename_dict_ = {
    "Product": "MATERIAL",
    "CrossPlantStatus": "MSTAE",
    "PdProductDescription": "Description",
    "H1_description": "H1",
    "H2_description": "H2",
    "H3_description": "H3",
    "H4_description": "H4",
    "H5_description": "H5"

}
master_data_col_list = ['Product', 'CrossPlantStatus',
                        'PdProductDescription', "LeadTime", 'ServiceLevel',
                        'H1_description', 'H2_description', 'H3_description',
                        "H4_description", "H5_description", 'NetWeight'
                        ]
master = collect_data(master_path, master_files)
master.drop(['H1', 'H2', 'H3', "H4", 'H5'], axis=1, inplace=True)
master.rename(columns=rename_dict_, inplace=True)
master.head().T

col_list=['BillingDocumentDateTimestamp',"DistributionChannel","ProductCode","BaseUnitOfMeasurementQuantity","SoldToParty","NetAmountInSGD"]
col_list
# sales_base_unit_path='../DATA_MILAN/S3BUCKER/SALES/JULY_SALES_DATA/'
sales_base_unit_path='../DATA_MILAN/S3BUCKER/SALES/JULY_11PM_UP_SALES/'
sales_base_unit_path_list=os.listdir(sales_base_unit_path)

sales=collect_data(sales_base_unit_path,sales_base_unit_path_list,col=col_list)
sales

sales.rename(columns={'BillingDocumentDateTimestamp':'BILLDATE',
     'DistributionChannel':'CHNL',
     'ProductCode':"MATERIAL",
     'BaseUnitOfMeasurementQuantity':"QTY_BASEUOM",
     'SoldToParty':"CUSTNO",
     'NetAmountInSGD':"SALES"},inplace=True)
sales.head()

dict_chanl_map={
    10:"B2B",
    20:"Export",
    21:"Re-Export",
    30:"Retail",
    31:'Ecomm'
}

sales['CHNL']=sales['CHNL'].astype(int)
sales['CHNL_NAME']=sales['CHNL'].map(dict_chanl_map)

sales['bill_date']=pd.to_datetime(sales['BILLDATE'],errors='coerce')

from datetime import datetime

sales['year_month']=sales['bill_date'].astype(str).str[:-3]


# sales['MATERIAL']=
sales["MATERIAL"]=pd.to_numeric(sales['MATERIAL'])

c=sales['MATERIAL'].isna()==False

sales=sales[c]
sales

sales["MATERIAL"]=sales['MATERIAL'].astype(int)
# sales['CUSTNO']=sales['CUSTNO'].astype(int)
sales.dtypes

sales["CUSTNO"]=pd.to_numeric(sales['CUSTNO'])


# sales["MATERIAL"]=sales['CUSTNO'].astype(int)
sales['CUSTNO']=sales['CUSTNO'].astype(int)


contract_path_files='./S3/CONTRACT_SALES/'
contract_path_files_path_list=os.listdir(contract_path_files)

rename_dict_={
    "ProductCode":'MATERIAL',
    "SoldToParty":"CUSTNO",
    "SalesContractValidityStartDateTimestamp":"contract_start_date",
    "SalesContractValidityEndDateTimestamp":"contract_end_date",

}
contract_sales=collect_data(contract_path_files,contract_path_files_path_list,col=list(rename_dict_.keys()))

rename_dict_={
    "ProductCode":'MATERIAL',
    "SoldToParty":"CUSTNO",
    "SalesContractValidityStartDateTimestamp":"contract_start_date",
    "SalesContractValidityEndDateTimestamp":"contract_end_date",

}
contract_sales.rename(columns=rename_dict_,inplace=True)

contract_sales['contract_start_date']=pd.to_datetime(contract_sales['contract_start_date'])
contract_sales['contract_end_date']=pd.to_datetime(contract_sales['contract_end_date'])

c=sales['CHNL_NAME']=="B2B"
sales_b2b=sales[c]
sales_b2b

contract_sales['MATERIAL']=contract_sales['MATERIAL'].astype(int)
contract_sales['CUSTNO']=contract_sales['CUSTNO'].astype(int)

sales_new = sales_b2b.copy()
for i in tqdm(range(contract_sales.shape[0])):
    contract_details = contract_sales.iloc[i, :]
    # print(contract_details)
    c1 = sales_new['MATERIAL'] == contract_details['MATERIAL']
    c2 = sales_new['bill_date'] >= contract_details['contract_start_date']
    c3 = sales_new['bill_date'] <= contract_details['contract_end_date']
    c4 = sales_new['CUSTNO'] == contract_details['CUSTNO']
    condition_list = (c1 & c2 & c3 & c4) == False
    if condition_list.sum() == 0:
        continue
    sales_new = sales_new[condition_list]
    #     display(filtered_sales)
    print(sales_new.shape)
#     sales_new=filtered_sales.copy
#     clear_output()

c=sales['CHNL_NAME']=="B2B"
non_b2b_sales=sales[~ c]


filtered_sales=pd.concat([non_b2b_sales,sales_new],axis=0)

filtered_sales['SALES']=filtered_sales['SALES'].astype(float)
filtered_sales['QTY_BASEUOM']=filtered_sales['QTY_BASEUOM'].astype(float).astype(int)


agg_dict={
    "QTY_BASEUOM":['sum','count'],
    "SALES":['sum']
}
groupby_data=filtered_sales.groupby(['MATERIAL','year_month',"CHNL_NAME"],as_index=False).agg(agg_dict)

groupby_data.columns=['MATERIAL','year_month','CHNL_NAME','QTY_BASEUOM_SUM',"QTY_BASEUOM_COUNT",'SALES_SUM']

groupby_data.to_csv('./INPUT_FILES/sales_groupby_31_07_24_NEW.csv',index=False,sep=',')
groupby_data
