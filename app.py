# import plotly_express as px
import streamlit as st
import pandas as pd
import numpy as np
from model import Model
from data_processing import Processing
import seaborn as sns

sns.set()
st.title("My Data Science Project")
df = pd.read_csv('house/house.csv')
mode = st.sidebar.radio("Mode", ["Visualization", "Model"])
if mode == "Visualization":
    st.header("Data Viewer")
    columns = list(df.columns)
    selections = st.multiselect(
        "Select the columns you want to display", columns, columns)
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
    st.text("Insert model here")

    # import the data
    df = pd.read_csv("train.csv")
    col = ['Street', 'HouseStyle', 'OverallQual', 'Functional', 'MiscFeature', 'SalePrice',
           'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'YrSold', 'BedroomAbvGr', 'KitchenAbvGr']
    test_col = ['Street', 'HouseStyle', 'OverallQual', 'Functional', 'MiscFeature', 'LotArea',
                'YearBuilt', '1stFlrSF', '2ndFlrSF', 'YrSold', 'BedroomAbvGr', 'KitchenAbvGr']

    df = df[col]

    cat_col = ['Street', 'HouseStyle',
               'OverallQual', 'Functional', 'MiscFeature']

    data = Processing(df, cat_col)
    df = data.main()

    #image = Image.open("house.png")
    st.title("Welcome to the House Price Prediction App")

    #st.image(image, use_column_width=True)

    rooms = st.multiselect(
        'Lets find out how many bedrooms?', df['BedroomAbvGr'].unique())
    # nationalities = st.multiselect('Show Player from Nationalities?', df['Nationality'].unique())
    new_df = df[(df['BedroomAbvGr'].isin(rooms))]

    # create figure using plotly express
    # fig = px.scatter(new_df, x . ='Overall',y='Age',color='Name')
    # # Plot!
    # st.plotly_chart(fig)
    # write dataframe to screen
    if not new_df.empty:
        st.write(new_df)

    st.write(
        "Now let's find out how much the prices when we choosing some parameters.")

    # input the numbers
    LotArea = st.slider("What is your square feet of Area?", int(
        df.LotArea.min()), int(df.LotArea.max()), int(df.LotArea.mean()))
    # OverallQual = st.sidebar("Quality of house?", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    OverallQual = st.slider("Quality of house?", int(df.OverallQual.min()), int(
        df.OverallQual.max()), int(df.OverallQual.mean()))
    bed = st.slider("How many bedrooms?", int(df.BedroomAbvGr.min()), int(
        df.BedroomAbvGr.max()), int(df.BedroomAbvGr.mean()))
    Year = st.slider("Which year built?", int(df.YearBuilt.min()), int(
        df.YearBuilt.max()), int(df.YearBuilt.mean()))

    # test_df = pd.read_csv('test.csv')
    # test_df = test_df[test_col]
    # test_data = Processing(test_df, cat_col)
    # test_df = test_data.main()
    # test_df = test_df.rename(columns = {'1stFlrSF': 'FirstFlrSF', '2ndFlrSF': 'SecondFlrSt'})
    # #modelling step
    #
    # pred = test_df.iloc[0]

    use_model = st.sidebar.selectbox(
        "Select Model?", ['LinearRegression', 'LassoRegression'])
    mod = Model(df, use_model)
    model, errors = mod.main()
    # import pdb; pdb.set_trace()
    # predictions = model.predict([[pred.Street, pred.HouseStyle, OverallQual, pred.Functional, pred.MiscFeature, LotArea,
    #                               Year, pred.FirstFlrSF, pred.SecondFlrSt, pred.YrSold, bed, pred.KitchenAbvGr]])
    predictions = model.predict([[np.int64(1), np.int64(2), OverallQual, int(6), int(-1), LotArea,
                                  Year, int(896), int(0), int(2010), bed, int(1)]])[0]

    # checking prediction house price
    if st.button("Run Prediction!"):
        st.header("Your house prices prediction is USD {}".format(
            int(predictions)))
        st.subheader("Your range of prediction is USD {} - USD {}".format(
            int(predictions-errors), int(predictions+errors)))
