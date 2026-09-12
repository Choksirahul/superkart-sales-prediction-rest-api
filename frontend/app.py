import streamlit as st
import pandas as pd
import requests

BACKEND_URL = "http://backend:7860"

st.title("SuperKart Sales Prediction")
st.write("Predict total product-store sales using the trained SuperKart regression model.")

st.subheader("Single Prediction")

product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.027, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_location = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])
product_id_char = st.selectbox("Product ID Prefix", ["FD", "NC", "DR"])
store_age = st.number_input("Store Age (Years)", min_value=0, value=16)
product_type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

payload = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Store_Age_Years": store_age,
    "Product_Type_Category": product_type_category
}

if st.button("Predict Sales", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
        if response.status_code == 200:
            prediction = response.json()["Predicted Product Store Sales Total"]
            st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
        else:
            st.error(response.json().get("error", "Prediction request failed."))
    except requests.RequestException as error:
        st.error(f"Unable to connect to the backend API: {error}")

st.subheader("Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file containing the required model features", type=["csv"])

if uploaded_file is not None and st.button("Predict Batch", type="primary"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={"file": uploaded_file},
            timeout=60
        )

        if response.status_code == 200:
            predictions = response.json()
            result_df = pd.DataFrame(
                list(predictions.items()),
                columns=["Row", "Predicted Product Store Sales Total"]
            )

            st.success("Batch predictions completed.")
            st.dataframe(result_df)

            st.download_button(
                "Download Predictions",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="superkart_batch_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(response.json().get("error", "Batch request failed."))
    except requests.RequestException as error:
        st.error(f"Unable to connect to the backend API: {error}")
