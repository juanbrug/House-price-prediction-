import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
from math import ceil, floor
import pickle
from PIL import Image


def app():
    df = pd.read_csv(r"data/house_price.csv")

    dropColumns = ["Id", "MSSubClass", "MSZoning", "Street", "LandContour", "Utilities", "LandSlope", "Condition1", "Condition2", "BldgType", "OverallCond", "RoofStyle",
                "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "Foundation", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "Heating", "Electrical", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath"] + ["SaleCondition", "SaleType", "YrSold", "MoSold", "MiscVal", "MiscFeature", "Fence", "PoolQC", "PoolArea", "ScreenPorch", "3SsnPorch", "EnclosedPorch", "OpenPorchSF", "WoodDeckSF", "PavedDrive", "GarageCond", "GarageQual", "GarageType", "FireplaceQu", "Functional", "KitchenAbvGr", "BedroomAbvGr"]

    droppedDf = df.drop(columns=dropColumns, axis=1)

    droppedDf.isnull().sum().sort_values(ascending=False)
    droppedDf["Alley"].fillna("NO", inplace=True)
    droppedDf["LotFrontage"].fillna(df.LotFrontage.mean(), inplace=True)
    droppedDf["GarageFinish"].fillna("NO", inplace=True)
    droppedDf["GarageYrBlt"].fillna(df.GarageYrBlt.mean(), inplace=True)
    droppedDf["BsmtQual"].fillna("NO", inplace=True)
    droppedDf["MasVnrArea"].fillna(0, inplace=True)
    droppedDf['MasVnrAreaCatg'] = np.where(droppedDf.MasVnrArea > 1000, 'BIG',
                                    np.where(droppedDf.MasVnrArea > 500, 'MEDIUM',
                                    np.where(droppedDf.MasVnrArea > 0, 'SMALL', 'NO')))

    droppedDf = droppedDf.drop(['SalePrice'], axis=1)
    inputDf = droppedDf.iloc[[0]].copy()

    for i in inputDf:
        if inputDf[i].dtype == "object":
            inputDf[i] = droppedDf[i].mode()[0]
        elif inputDf[i].dtype == "int64" or inputDf[i].dtype == "float64":
            inputDf[i] = droppedDf[i].mean()

    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    # load the model weights and predict the target
    modelName = r"trained_model.model"
    loaded_model = pickle.load(open(modelName, 'rb'))

    # %% STREAMLIT DISPLAYED INFO
    st.title("House Prices Prediction")
    st.write("##### Model for house prices prediction from the Kaggle Ames, Iowa Housing Dataset.")
    #st.write("The dataset contains the following features:")
    #st.write("OverallQual: Overall quality of the house")
    #st.write("GrLivArea: Above grade (ground) living area square feet")
    #st.write("GarageCars: Number of garage cars")
    #st.write("TotalBsmtSF**: Total square feet of basement area")
    #st.write("FullBath: Number of full baths")
    #st.write("YearBuilt: Year house was built")
    #st.write("TotRmsAbvGrd: Total number of rooms above grade (excluding bathrooms and closets")
    #st.write("Fireplaces: Number of fireplaces")
    #st.write("BedroomAbvGr: Number of bedrooms above grade")
    #st.write("GarageYrBlt: Year garage was built")
    #st.write("LowQualFinSF: Lowest quality finished square feet")
    #st.write("LotFrontage: Lot frontage square feet")
    #st.write("MasVnrArea: Masonry veneer square feet")    
    #st.write("WoodDeckSF: Square feet of wood deck area")
    #st.write("penPorchSF: Open porch square feet")
    #st.write("EnclosedPorch: Enclosed porch square feet")
    #st.write("3SsnPorch: Three season porch square feet")
    #st.write("ScreenPorch: Screen porch square feet")
    #st.write("PoolArea: Pool square feet")
    #st.write("MiscVal: Miscellaneous value")
    #st.write("MoSold: Month house was sold")
    #st.write("YrSold: Year house was sold")
    #st.write("SalePrice: Sale price")
    image = Image.open('./data/features.png')
    st.image(image)

    st.sidebar.title("Model Parameters")
    st.sidebar.write("### Feature importance of model (20 most important features")
    
    expander= st.sidebar.expander("Click for Model Features ")
    expander.write("## Features")
    
    # Get Feature importance of model, 20 most important features for the prediction
    featureImportances = pd.Series(loaded_model.feature_importances_,index = droppedDf.columns).sort_values(ascending=False)[:20]
    
    inputDict = dict(inputDf)

    for idx, i in enumerate(featureImportances.index):
        if droppedDf[i].dtype == "object":
            variables = droppedDf[i].drop_duplicates().to_list()
            inputDict[i] = expander.selectbox(i, options=variables, key=idx)
        elif droppedDf[i].dtype == "int64" or droppedDf[i].dtype == "float64":
            inputDict[i] = expander.slider(i, ceil(droppedDf[i].min()),
                                                floor(droppedDf[i].max()), int(droppedDf[i].mean()), key=idx)
        else:
            expander.write(i)


    for key, value in inputDict.items():
        inputDf[key] = value

    obj_feat = list(inputDf.loc[:, inputDf.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        inputDf[feature] = inputDf[feature].astype('category')

    prediction = loaded_model.predict(inputDf)

    st.write("###### Predicted price of the house based on selected features: $", prediction.item())

    st.markdown("------")

    


app()
