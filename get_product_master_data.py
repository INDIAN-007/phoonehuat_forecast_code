import os

import numpy as np

from static_variables import product_hierarchy_path
from utility import collect_data


def get_product_master_data():
    product_hierarchy_files = os.listdir(product_hierarchy_path)

    col = ['Product', 'H1_description', "H2_description", "H3_description", "H4_description", "H5_description",
           'PdProductDescription', 'CrossPlantStatus', 'NetWeight']
    sku_master_data = collect_data(product_hierarchy_path, product_hierarchy_files, col=col)
    sku_master_data.rename(columns={'Product': "MATERIAL", 'H1_description': "H1", "H2_description": "H2"
        , "H3_description": 'H3', "H4_description": "H4", "H5_description": "H5", 'NetWeight': "KG",
                                    "CrossPlantStatus": "MSTAE"}, inplace=True)
    sku_master_data = sku_master_data.astype({"MATERIAL": np.int32})
    return sku_master_data
