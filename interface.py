import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import random


st.markdown(
    """
    <style>
    /* App background and default text color */
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    /* This rule targets text elements WITHOUT affecting widget internals */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #000000 !important;
    }
    
    /* --- FINAL FIX: Targets the text inside the dropdown menu options --- */
    li[role="option"] {
        color: white !important;
    }

    /* Buttons */
    button, .stButton>button {
        background-color: #00aaff !important;  /* button background */
        color: #FFFFFF !important;             /* button text */
        font-weight: bold;
        border-radius: 8px;
        outline: none !important;              /* remove focus outline */
        box-shadow: none !important;           /* remove shadow */
        border: none !important;               /* remove any default border */
    }

    /* Button hover effect */
    button:hover, .stButton>button:hover {
        background-color: #0077aa !important;
        color: #ffffff !important;
    }

    /* CSS for the price estimate box */
    .price-box {
        background-color: #e0f7fa; /* Light cyan background */
        border: 2px solid #00aaff; /* Border color matching the button */
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .price-header {
        font-size: 22px;
        font-weight: bold;
        color: #004d66 !important; /* A darker shade for the header */
        margin-bottom: 10px;
    }
    .price-text {
        font-size: 32px;
        font-weight: bold;
        color: #0077aa !important; /* A bright, clear blue for the price */
    }

    </style>
    """,
    unsafe_allow_html=True
)




@st.cache_data
def load_artifacts():
    high_model = xgb.XGBRegressor()
    high_model.load_model('xgb_model_high.json')

    zip_stats = joblib.load('zip_stats.joblib')
    city_stats = joblib.load('city_stats.joblib')
    model_columns = joblib.load('model_columns.joblib')
    return high_model, zip_stats, city_stats, model_columns

high_model , zip_stats, city_stats, model_columns = load_artifacts()


def extract_description_features(description):
    description = str(description).lower()
    pool_keywords = ['pool', 'swimming', 'poolside','in-ground', 'heated pool', 'pool area', 'pool deck', 'spa', 'jacuzzi', 'hot tub']
    remodel_keywords = ['remodeled', 'renovated', 'updated', 'newly done', 'fully upgraded', 'modernized', 'recently renovated', 'new finishes', 'newly remodeled', 'newly renovated','new', 'new construction', 'newly built']
    roof_keywords = ['new roof', 'roof replaced', 'recent roof']
    kitchen_keywords = ['granite', 'quartz', 'stainless steel', 'new kitchen', 'updated kitchen', 'gourmet kitchen','chef\'s kitchen', 'modern kitchen', 'luxury kitchen', 'kitchen remodel', 'kitchen renovation', 'kitchen upgrade']

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

    if subtype in remove_group:
        return None
    if subtype in condo_group:
        return 'Condominium'
    if subtype in sfr_group:
        return 'Single Family Residence'
    if subtype in multi_family_group:
        return 'Multi Family'
    return 'Other'

def engineer_features_single(input_df, zip_stats, city_stats):
    df = input_df.copy()
    # Ratios
    df['BathBedRatio'] = (df['Beds'] / df['Beds']).replace([np.inf, -np.inf], 0).fillna(0)
    df['HouseLotRatio'] = (df['SquareFootage'] / df['PropertyLot_Square_footage']).replace([np.inf, -np.inf], 0).fillna(0)
    # ZIP features
    if 'ZIP' in df.columns and zip_stats is not None:
        df = df.merge(zip_stats, on='ZIP', how='left')
        for col in zip_stats.columns:
            if col != 'ZIP':
                df[col] = df[col].fillna(zip_stats[col].mean())
    # City features
    if 'City' in df.columns and city_stats is not None:
        df = df.merge(city_stats, on='City', how='left')
        for col in city_stats.columns:
            if col != 'City':
                df[col] = df[col].fillna(city_stats[col].mean())
    # PropertyCategory dummies
    if 'PropertyCategory' in df.columns:
        df = pd.get_dummies(df, columns=['PropertyCategory'], drop_first=True)
    # Align with training columns
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

# -------------------------------
# Streamlit Interface
# -------------------------------
st.set_page_config(page_title="The Price-Timator", layout="centered")
st.title("Sahi Sahi Bhav Lagao Bhaya")

# --- Initialize session state ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'address' not in st.session_state:
    st.session_state.address = ""

# --- Step 1: Address input ---
if st.session_state.step == 1:
    st.subheader("Enter Property Address")
    address = st.text_input("Full Property Address)", value=st.session_state.address)
    if st.button("Next"):
        if address.strip() == "":
            st.warning("Please enter a valid address")
        else:
            st.session_state.address = address
            st.session_state.step = 2
            st.rerun() # Use rerun to smoothly transition to the next step

# --- Step 2: Property Details ---
if st.session_state.step == 2:
    st.subheader("Step 2: Enter Property Details")
    with st.form("property_details_form"):
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
            City = st.text_input("City", value="Miami")
            ZIP = st.text_input("ZIP Code", value="33101")
            PropertySubType = st.selectbox("Property SubType", ["SingleFamilyResidence","Condominium","Townhouse","MultiFamily","Duplex","Residential"])

        Description = st.text_area("Property Description", height=100, help="Mention pool, remodel, kitchen upgrades etc.")
        submitted = st.form_submit_button("Get Estimate")

        if submitted:
            try:
                # --- Convert inputs and validate ---
                Beds = int(Beds_str)
                Baths = float(Baths_str)
                HalfBaths = float(HalfBaths_str)
                SquareFootage = int(SquareFootage_str)
                PropertyLot_Square_footage = int(PropertyLot_Square_footage_str)
                GarageSpaces = int(GarageSpaces_str)
                YearBuilt = int(YearBuilt_str)

                # --- Process internal features ---
                prop_features = extract_description_features(Description)
                PropertyCategory = map_property_category(PropertySubType)
                IsAttached = 1 if PropertyCategory == 'Condominium' else 0
                PropertyAge = 2025 - YearBuilt  # Using a fixed current year for consistency

                # Build dataframe
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

                # Engineer features
                processed_df = engineer_features_single(input_df, zip_stats, city_stats)

                # Predict
                predicted = np.expm1(high_model.predict(processed_df))[0]
                
                # Create a price range
                high_start = predicted + 60000
                high_end = predicted + random.uniform(95000, 100000)
                
                # Display the result in the new styled box
                price_range_str = f"${high_start:,.0f} - ${high_end:,.0f}"

                st.markdown(f"""
                <div class="price-box">
                    <p class="price-header">Your Estimated Property Value Is:</p>
                    <p class="price-text">{price_range_str}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.balloons()

            except ValueError:
                st.error("Please enter valid numbers for all numerical fields (Bedrooms, Bathrooms, SqFt, etc.).")
            except Exception as e:
                st.error(f"An error occurred: {e}")