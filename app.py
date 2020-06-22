import streamlit as st
import pandas as pd
import numpy as np
from model import Model
from data_processing import Processing
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


sns.set()
st.title("My Data Science Project")
df = load_data('house/house.csv')

mode = st.sidebar.radio("Mode", ["Visualization", "Model"])
if mode == "Visualization":
    st.header("Data Viewer")
    columns = list(df.columns)
    selections = st.multiselect(
        "Select the columns you want to display", columns)
    if selections:
        st.dataframe(df[selections].drop_duplicates())
    else:
        st.dataframe(df)

    st.header('Visualization')
    xaxis = st.selectbox("X-Axis:", columns, 0)
    yaxis = st.selectbox("Y-Axis:", columns, 1)
    hue = st.selectbox("Color:", columns, 2)
    sns.scatterplot(x=xaxis, y=yaxis, data=df, hue=hue)
    plt.title(f"{xaxis} vs {yaxis}")
    st.pyplot()
else:
    st.header("Machine Learning Model")

    cat_col = ['Street', 'HouseStyle',
               'OverallQual', 'Functional', 'MiscFeature']

    data = Processing(df, cat_col)
    processed_df = data.main()

    st.title("Welcome to the House Price Prediction App")

    st.write(
        "Now let's find out how much the prices when we choosing some parameters.")

    # input the numbers
    LotArea = st.slider(
        "What is your square feet of Area?",
        int(processed_df.LotArea.min()),
        int(processed_df.LotArea.max()),
        int(processed_df.LotArea.mean())
    )
    OverallQual = st.number_input(
        "Quality of house?",
        int(processed_df.OverallQual.min()),
        int(processed_df.OverallQual.max()),
        int(processed_df.OverallQual.mean())
    )
    bed = st.slider(
        "How many bedrooms?",
        int(processed_df.BedroomAbvGr.min()),
        int(processed_df.BedroomAbvGr.max()),
        int(processed_df.BedroomAbvGr.mean())
    )
    Year = st.slider(
        "Which year built?",
        int(processed_df.YearBuilt.min()),
        int(processed_df.YearBuilt.max()),
        int(processed_df.YearBuilt.mean())
    )

    results = pd.DataFrame(columns=['model_name', 'alpha', 'errors'])

    model_name = st.sidebar.selectbox(
        "Select Model?", ['LinearRegression', 'LassoRegression'])

    alpha = 0
    alpha = st.number_input('Input required alpha here:')

    if model_name == "LinearRegression":
        mod = Model(processed_df, model_name, alpha)
    else:
        mod = Model(processed_df, model_name, alpha)
    model, errors = mod.main()

    predictions = model.predict([[np.int64(1), np.int64(2), OverallQual, int(6), int(-1), LotArea,
                                  Year, int(896), int(0), int(2010), bed, int(1)]])[0]

    # checking prediction house price
    if st.button("Run Prediction!"):
        st.header("Your house prices prediction is USD {}".format(
            int(predictions)))
        st.subheader("Your range of prediction is USD {} - USD {}".format(
            int(predictions-errors), int(predictions+errors)))

        results = results.append(
            {'model_name': model_name, 'alpha': alpha, 'errors': errors}, ignore_index=True)
        st.table(results)
