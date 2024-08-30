# Horizon & Loop Back and Forecast editable section
from datetime import datetime

production = True
# back_testing=True

heirarchy_list = [["H2", "H3", "H4"], ["H2", "H3", "KG"]]

loop_back_time = 3
max_horizon = 6
year = 2023
month = 11
date = 1
current_date = datetime(year, month, date)

# MASTER FILES PATH
product_hierarchy_path = './INPUT_FILES/product_master_files/'
sales_file_path = './INPUT_FILES/sales_groupby_31_07_24.csv'
main_path = './'

filter_a_b_category_file_path = './INPUT_FILES/filter_BIN_.csv'

"""
RULES
rule1=enter the channel seperated by underscore
rule2=Do not keep underscore character in the channel name
"""

categories_for_sku = {
    # "Dairy": ['Export_Re-Export', "Retail_Ecomm", 'B2B'],
    # "Bakery": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Fruits": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Beverage": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Cheese": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Chocolates": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Fat & Oil": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Filling & Jam": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Flour, Grain & Flakes": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Grocery": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    # "Meat": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    "Non Food": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    "Nuts, Seeds & Beans": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    "PHD": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
    "Seafood": ["B2B", 'Export_Re-Export', "Retail_Ecomm"],
}

categories_for_combo = {
    # "Dairy": ["Retail_Ecomm", 'B2B'],
    # "Bakery": ["B2B", "Retail_Ecomm"],
    # "Fruits": ["B2B", "Retail_Ecomm"],
    # "Beverage": ["B2B", "Retail_Ecomm"],
    # "Cheese": ["B2B", "Retail_Ecomm"],
    # "Chocolates": ["B2B", "Retail_Ecomm"],
    # "Fat & Oil": ["B2B", "Retail_Ecomm"],
    # "Filling & Jam": ["B2B", "Retail_Ecomm"],
    # "Flour, Grain & Flakes": ["B2B", "Retail_Ecomm"],
    # "Grocery": ["B2B", "Retail_Ecomm"],
    # "Meat": ["B2B", "Retail_Ecomm"],
    "Non Food": ["B2B", "Retail_Ecomm"],
    "Nuts, Seeds & Beans": ["B2B", "Retail_Ecomm"],
    "PHD": ["B2B", "Retail_Ecomm"],
    "Seafood": ["B2B", "Retail_Ecomm"],
}



