import pandas as pd
from langchain_core.documents import Document

def dataconverter():

    product_data=pd.read_csv(r"G:\flipkart\data\flipkart_updated.csv")
    
    data=product_data(["ProductName","Description"])

    product_list= []

# iterator over the dataframe

    for index, row in data.iterrows():
        object = {
            "product_name": row["ProductName"],
            "description" : row['combined']

        }
    # append the object to the product list

    product_list.append(object)
    docs = []

    for object in product_list:
        metadata = {"product_name": object["product_name"]}
        page_content = object["description"]

        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    return docs




