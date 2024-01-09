import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder

def train_model(df, alpha=1.0):
    le = LabelEncoder()
    df['mainroad'] = le.fit_transform(df['mainroad'])
    df['guestroom'] = le.fit_transform(df['guestroom'])
    df['basement'] = le.fit_transform(df['basement'])
    df['hotwaterheating'] = le.fit_transform(df['hotwaterheating'])
    df['airconditioning'] = le.fit_transform(df['airconditioning'])
    df['prefarea'] = le.fit_transform(df['prefarea'])
    df['furnishingstatus'] = le.fit_transform(df['furnishingstatus'])

    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)

    return model, le

def predict(model, le, input_data):
    input_data['mainroad'] = le.fit_transform(input_data['mainroad'])
    input_data['guestroom'] = le.fit_transform(input_data['guestroom'])
    input_data['basement'] = le.fit_transform(input_data['basement'])
    input_data['hotwaterheating'] = le.fit_transform(input_data['hotwaterheating'])
    input_data['airconditioning'] = le.fit_transform(input_data['airconditioning'])
    input_data['prefarea'] = le.fit_transform(input_data['prefarea'])
    input_data['furnishingstatus'] = le.fit_transform(input_data['furnishingstatus'])

    prediction = model.predict(input_data)
    return prediction

def main():
    title_text = "HOUSE PRICE PREDICTION(INDIA)"


    styled_title = f'<h1 style="margin-bottom: 10px; color: red;">{title_text}</h1>'


    st.markdown(styled_title, unsafe_allow_html=True)

    
    


    file_path = r"C:\Users\jagat\OneDrive\Desktop\CODE CASA\TASK 2\indiahousepricing.csv"
    df = pd.read_csv(file_path)

    alpha = st.sidebar.slider("LASSO STRENGTH", min_value=0.01, max_value=10.0, step=0.01, value=1.0)
    model, le = train_model(df, alpha)

    st.sidebar.title("USER VALUES")

    area = st.sidebar.slider("Area", min_value=int(df['area'].min()), max_value=int(df['area'].max()), step=100, value=int(df['area'].mean()))
    bedrooms = st.sidebar.slider("Bedrooms", min_value=int(df['bedrooms'].min()), max_value=int(df['bedrooms'].max()), step=1, value=int(df['bedrooms'].mean()))
    bathrooms = st.sidebar.slider("Bathrooms", min_value=int(df['bathrooms'].min()), max_value=int(df['bathrooms'].max()), step=1, value=int(df['bathrooms'].mean()))
    stories = st.sidebar.slider("Stories", min_value=int(df['stories'].min()), max_value=int(df['stories'].max()), step=1, value=int(df['stories'].mean()))
    mainroad = st.sidebar.selectbox("Main Road", ['YES', 'NO'], index=0)
    guestroom = st.sidebar.selectbox("Guest Room", ['YES', 'NO'], index=0)
    basement = st.sidebar.selectbox("Basement", ['YES', 'NO'], index=0)
    hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ['YES', 'NO'], index=0)
    airconditioning = st.sidebar.selectbox("Air Conditioning", ['YES', 'NO'], index=0)
    parking = st.sidebar.slider("Parking", min_value=int(df['parking'].min()), max_value=int(df['parking'].max()), step=1, value=int(df['parking'].mean()))
    prefarea = st.sidebar.selectbox("Preferred Area", ['YES', 'NO'], index=0)
    furnishingstatus = st.sidebar.selectbox("Furnishing Status", ['FURNISHED', 'SEMI FURNISHED', 'UNFURNISHED'], index=0)

    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'stories': [stories],
        'mainroad': [mainroad],
        'guestroom': [guestroom],
        'basement': [basement],
        'hotwaterheating': [hotwaterheating],
        'airconditioning': [airconditioning],
        'parking': [parking],
        'prefarea': [prefarea],
        'furnishingstatus': [furnishingstatus]
    })

    prediction = predict(model, le, input_data)

    st.subheader("Prediction Result:")
    st.write(f"The predicted price for the given input is: ")
    st.markdown(f"<h6 style='color: red;'>{prediction[0]:,.2f} INR</h6>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
