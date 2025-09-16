import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import random
import re

st.set_page_config(page_title="The Price-Timator", layout="centered")
st.title("Know what your property is worth in Miami MLS with The Price-Timator!")

# css style
st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff !important; color: #000000 !important; }
    h1, h2, h3, h4, h5, h6, p, span, label { color: #000000 !important; }
    li[role="option"] { color: white !important; }
    button, .stButton>button {
        background-color: #00aaff !important;
        color: #FFFFFF !important;
        font-weight: bold;
        border-radius: 8px;
        border: none !important;
    }
    button:hover, .stButton>button:hover { background-color: #0077aa !important; }
    .price-box {
        background-color: #e0f7fa;
        border: 2px solid #00aaff;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .price-header { font-size: 22px; font-weight: bold; color: #004d66 !important; margin-bottom: 10px; }
    .price-text { font-size: 32px; font-weight: bold; color: #0077aa !important; }
    </style>
    """,
    unsafe_allow_html=True
)

#loading model
@st.cache_data
def load_artifacts():
    high_model = xgb.XGBRegressor()
    high_model.load_model('xgb_model_high.json')
    zip_stats = joblib.load('zip_stats.joblib')
    city_stats = joblib.load('city_stats.joblib')
    model_columns = joblib.load('model_columns.joblib')
    return high_model, zip_stats, city_stats, model_columns

high_model, zip_stats, city_stats, model_columns = load_artifacts()

#feature engineering functions
def extract_zip(address: str):
    """Extract 5-digit ZIP from address string"""
    match = re.search(r"\b\d{5}\b$", address)
    return match.group(0) if match else ""


def extract_description_features(description):
    description = str(description).lower()
    pool_keywords = ['pool', 'swimming', 'poolside','in-ground', 'heated pool', 'spa', 'jacuzzi', 'hot tub']
    remodel_keywords = ['remodeled', 'renovated', 'updated', 'newly done', 'modernized', 'recently renovated', 'newly remodeled', 'newly built']
    roof_keywords = ['new roof', 'roof replaced', 'recent roof']
    kitchen_keywords = ['granite', 'quartz', 'stainless steel', 'new kitchen', 'updated kitchen', 'gourmet kitchen','chef\'s kitchen', 'luxury kitchen']

    return {
        'HasPrivatePool': int(any(k in description for k in pool_keywords)),
        'IsRemodeled': int(any(k in description for k in remodel_keywords)),
        'HasNewRoof': int(any(k in description for k in roof_keywords)),
        'HasUpgradedKitchen': int(any(k in description for k in kitchen_keywords))
    }

def map_property_category(subtype):
    condo_group = ['Condominium', 'Townhouse', 'Apartment', 'Villa', 'StockCooperative']
    sfr_group = ['SingleFamilyResidence', 'MobileHome']
    multi_family_group = ['MultiFamily', 'Duplex', 'Residential']
    remove_group = ['Timeshare', 'HotelMotel', 'BoatSlip', 'Other', 'Office', 'Industrial']
    if subtype in remove_group: return None
    if subtype in condo_group: return 'Condominium'
    if subtype in sfr_group: return 'Single Family Residence'
    if subtype in multi_family_group: return 'Multi Family'
    return 'Other'

def engineer_features_single(input_df, zip_stats, city_stats):
    df = input_df.copy()
    df['BathBedRatio'] = (df['Baths'] / df['Beds']).replace([np.inf, -np.inf], 0).fillna(0)
    df['HouseLotRatio'] = (df['SquareFootage'] / df['PropertyLot_Square_footage']).replace([np.inf, -np.inf], 0).fillna(0)

    if 'ZIP' in df.columns and zip_stats is not None:
        df = df.merge(zip_stats, on='ZIP', how='left')
        for col in zip_stats.columns:
            if col != 'ZIP':
                df[col] = df[col].fillna(zip_stats[col].mean())

    if 'City' in df.columns and city_stats is not None:
        df = df.merge(city_stats, on='City', how='left')
        for col in city_stats.columns:
            if col != 'City':
                df[col] = df[col].fillna(city_stats[col].mean())

    if 'PropertyCategory' in df.columns:
        df = pd.get_dummies(df, columns=['PropertyCategory'], drop_first=True)

    df = df.reindex(columns=model_columns, fill_value=0)
    return df


#streamlit form


address = st.text_input("Full Property Address", value="")

st.subheader("Enter Property Details")
with st.form("property_form"):
    

    col1, col2 = st.columns(2)
    with col1:
        Beds_str = st.text_input("Bedrooms", value="3")
        Baths_str = st.text_input("Bathrooms", value="2")
        HalfBaths_str = st.text_input("Half Bathrooms", value="0")
        SquareFootage_str = st.text_input("Living Area (sqft)", value="1500")
    with col2:
        PropertyLot_Square_footage_str = st.text_input("Lot Size (sqft)", value="5000")
        GarageSpaces_str = st.text_input("Garage Spaces", value="1")
        YearBuilt_str = st.text_input("Year Built", value="2000")
        City = st.selectbox("City", ["Miami", "Miami Beach", "Hialeah", "Coral Gables", "North Miami", 
                                     "Aventura", "Sunny Isles Beach", "Doral", "Kendall", "Pinecrest", "Cutler Bay", "Homestead", "Hollywood"])

        # Auto extract ZIP
        extracted_zip = extract_zip(address)
        ZIP = st.text_input("ZIP Code", value=extracted_zip if extracted_zip else "")

        PropertySubType = st.selectbox(
            "Property Sub Type",
            ["Single Family Residence","Condominium","Townhouse","MultiFamily","Duplex","Residential"]
        )

    Description = st.text_area("Property Description", height=100, help="Mention pool, remodel, kitchen upgrades etc.")
    submitted = st.form_submit_button("Get Estimate")

#predictions
if submitted:
    try:
        Beds = int(Beds_str)
        Baths = float(Baths_str)
        HalfBaths = float(HalfBaths_str)
        SquareFootage = int(SquareFootage_str)
        PropertyLot_Square_footage = int(PropertyLot_Square_footage_str)
        GarageSpaces = int(GarageSpaces_str)
        YearBuilt = int(YearBuilt_str)

        prop_features = extract_description_features(Description)
        PropertyCategory = map_property_category(PropertySubType)
        IsAttached = 1 if PropertyCategory == 'Condominium' else 0
        PropertyAge = 2025 - YearBuilt

        input_df = pd.DataFrame([{
            'Beds': Beds,
            'Baths': Baths,
            'HalfBaths': HalfBaths,
            'SquareFootage': SquareFootage,
            'PropertyLot_Square_footage': PropertyLot_Square_footage,
            'GarageSpaces': GarageSpaces,
            'PropertyAge': PropertyAge,
            'City': City,
            'ZIP': ZIP.zfill(5),
            'PropertyCategory': PropertyCategory,
            'IsAttached': IsAttached,
            **prop_features
        }])

        processed_df = engineer_features_single(input_df, zip_stats, city_stats)
        predicted = np.expm1(high_model.predict(processed_df))[0]

        high_start = predicted
        high_end = predicted + random.uniform(50000, 60000)

        price_range_str = f"${high_start:,.0f} - ${high_end:,.0f}"
        st.markdown(f"""
        <div class="price-box">
            <p class="price-header">Your Estimated Property Value Is:</p>
            <p class="price-text">{price_range_str}</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()

    except ValueError:
        st.error("Please enter valid numbers for all numerical fields.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
