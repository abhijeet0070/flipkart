import pandas as pd
from langchain.schema import Document

def dataconverter():
    # Load your dataset
    product_data = pd.read_csv(r"G:\flipkart\data\flipkart_updated.csv")

    # Only keep needed columns
    product_data = product_data[["ProductName", "Description"]]

    # ✅ Create 'combined' column
    product_data["combined"] = product_data["ProductName"] + " - " + product_data["Description"]

    # Convert to LangChain documents
    docs = []
    for idx, row in product_data.iterrows():
        doc = Document(
            page_content=row["combined"],
            metadata={"row": idx}
        )
        docs.append(doc)

    return docs
