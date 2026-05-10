import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error,accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

st.set_page_config(page_title="ML Dasboard",layout="wide")
st.title("ML Dashboard(Regression+Classification)")


menu=st.sidebar.selectbox(
    "Choose Section",
    ["Upload Data","Train Model","Predict"]
)
show_data=st.sidebar.checkbox("Show Dataset")

if "df" not in st.session_state:
    st.session_state.df=None
if "reg_model" not in st.session_state:
    st.session_state.reg_model=None
if "clf_model" not in st.session_state:
    st.session_state.clf_model=None

if menu=="Upload Data":
        st.header("Upload Dataset") 
        file=st.file_uploader("Upload CSV file",type=["csv"])
        if file:
             df=pd.read_csv(file)
             st.session_state.df=df
             st.success("Data Loaded Successfully")
             if show_data:
                 st.dataframe(df)

elif menu=="Train Model":
     st.header("Train ML Models")

     if st.session_state.df is None:
        st.warning("Please upload dataset first")
     else:
          df=st.session_state.df
          st.write("Dastaset Preview")
          st.dataframe(df.head())
          target_reg=st.selectbox("Select Regression Target",df.columns) 
          target_clf=st.selectbox("Select Classification Target",df.columns) 
          X=df.drop([target_reg,target_clf],axis=1,errors='ignore')
          X=pd.get_dummies(X)
          y_reg=df[target_reg]
          y_clf=df[target_clf]    
          st.session_state.columns=X.columns
          X_train,X_test,y_train_reg,y_test_reg=train_test_split(X,y_reg,test_size=0.2,random_state=42)
          y_train_clf=y_clf.loc[y_train_reg.index]
          y_test_clf=y_clf.loc[y_test_reg.index]          
         
          if st.button("Train Models"):
               
            reg_model=RandomForestRegressor()
            reg_model.fit(X_train,y_train_reg)
            reg_pred=reg_model.predict(X_test)
            mse=mean_squared_error(y_test_reg,reg_pred)

            clf_model=RandomForestClassifier()
            clf_model.fit(X_train,y_train_clf)
            clf_pred=clf_model.predict(X_test)
            acc=accuracy_score(y_test_clf,clf_pred)

            st.session_state.reg_model=reg_model
            st.session_state.clf_model=clf_model
            st.success("Models Trained Successfully")
            st.metric("Regression MSE",round(mse,2))
            st.metric("Classification Accuracy",round(acc*100,2))

            fig,ax=plt.subplots()
            ax.plot(y_test_reg.values[:20],label="Actual")
            ax.plot(reg_pred[:20],label="Predicted")
            ax.legend()
            st.pyplot(fig)
            joblib.dump(reg_model,"reg_model.pkl")
            joblib.dump(clf_model,"clf_model.pkl")
            st.info("Models saved as .pkl files")


elif menu=="Predict":
    st.header("Make Predictions")
    if st.session_state.reg_model is None:
       st.warning("Train model first")
    else:
        df=st.session_state.df
        X=df.drop(df.columns[-2:],axis=1,errors="ignore") 
        X=pd.get_dummies(X)
        input_data={} 
        st.write("Enter input values:")
        for col in X.columns:
            input_data[col]=st.number_input(col,value=0.0)
        input_df=pd.DataFrame([input_data])
        input_df=input_df.reindex(columns=st.session_state.columns,fill_value=0)

        if st.button("Predict"):
            reg_result=st.session_state.reg_model.predict(input_df)[0]
            clf_result=st.session_state.clf_model.predict(input_df)[0]

            st.success(f"Predicted Value:{reg_result}")
            st.success(f"Classification:{clf_result}")    
