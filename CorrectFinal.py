import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import streamlit as st
import pickle
import numpy as np
from wordcloud import WordCloud
import pandas as pd
from streamlit_option_menu import option_menu
with st.sidebar:
    selection=option_menu("Choose The Prediction",['Fraud Claim Detection','Customer Segmenation','Sentiment Analysis','Insurance Risk_Score','Insurance Claim_Amount'],
                          menu_icon='cash-stack',
                          icons=['person-fill','people-fill','graph-up','cash','currency-dollar'],
                          default_index=0)
with open("C:/Users/sakth/fraudcorrect.pkl", "rb") as file:
    model = pickle.load(file)
with open("C:/Users/sakth/customersegmentcorrect.pkl",'rb')as file:
    model1=pickle.load(file)
with open("C:/Users/sakth/insuranceClassification1.pkl",'rb')as file:
    model2=pickle.load(file)
with open("C:/Users/sakth/insuranceregression1.pkl",'rb') as file:
    model3=pickle.load(file)
if selection=="Fraud Claim Detection":
    st.markdown(
        f"""
            <style>
               .stApp {{
                background-color:#87ceeb
            }}
               .main-content {{
                    padding: 20px;
                    border-radius: 10px;
                    background-color: 	rgba(160, 230, 255, 0.6); /* Add slight transparency */
                    color: black;
                    text-align: center;
                }}
            </style>
            """,
        unsafe_allow_html=True)
    st.markdown("<div class='main-content'><h1>Fraud Prediction</h1><p>Secure & Reliable</p></div>",
                unsafe_allow_html=True)
    st.image("C:/Users/sakth\Downloads\efg-hermes-fraud-prevention-codix.jpg")
    col1,col2=st.columns(2)
    with col1:
      Claim_Amount = st.number_input("Claim Amount")
      Suspicious_Flags = st.number_input("Suspicious Flags")
      Claim_Type_Auto = st.number_input("Claim Type Auto")
    with col2:
       Claim_Type_Home = st.number_input("Claim Type Home")
       Claim_Type_Life = st.number_input("Claim Type Life")
       Claim_Type_Medical = st.number_input("Claim Type Medical")
    inputdata = np.array([
    Claim_Amount, Suspicious_Flags, Claim_Type_Auto,
    Claim_Type_Home, Claim_Type_Life, Claim_Type_Medical]).reshape(1, -1)
    feature_names = model.feature_names_in_
    inputdata_df = pd.DataFrame(inputdata, columns=feature_names)
    if st.button("PREDICT"):
        prediction = model.predict(inputdata_df)
        if prediction[0] == 1:
           st.error("🚨 Claim is Fraudulent!")
        else:
           st.success("✅ Claim is Genuine")
elif selection=='Customer Segmenation':
    st.markdown(
        f"""
                <style>
                   .stApp {{
                    background-color:#FFD700

                }}
                   .main-content {{
                        padding: 20px;
                        border-radius: 10px;
                        background-color: 	rgb(245, 222, 179); /* Add slight transparency */
                        color: black;
                        text-align: center;
                    }}
                </style>
                """,
        unsafe_allow_html=True)
    st.markdown("<div class='main-content'><h1>Customer Segmentation </h1><p>Easy Analysis</p></div>",
                unsafe_allow_html=True)
    st.image("C:/Users/sakth\Downloads\segmentation-for-growth.jpg")
    Policy_Count=st.number_input("Policy_Count")
    Claim_Frequency=st.number_input("Claim_Frequency")
    Policy_Upgrades=st.number_input("Policy_Upgrades")
    Kmeans_Cluster=st.number_input(" Kmeans_Cluster")
    inputdata1 = np.array([Policy_Count,Claim_Frequency,Policy_Upgrades, Kmeans_Cluster
        ]).reshape(1, -1)
    feature_names = model1.feature_names_in_
    inputdata_df1 = pd.DataFrame(inputdata1, columns=feature_names)
    if st.button("PREDICT"):
        prediction = model1.predict(inputdata_df1)
        st.success(*prediction)
elif selection=='Sentiment Analysis':
    st.markdown(
        f"""
                    <style>
                       .stApp {{
                        background-color:#D3D3D3

                    }}
                       .main-content {{
                            padding: 20px;
                            border-radius: 10px;
                            background-color: 	rgba(50, 50, 50, 0.6); /* Add slight transparency */
                            color: black;
                            text-align: center;
                        }}
                    </style>
                    """,
        unsafe_allow_html=True)
    st.markdown("<div class='main-content'><h1>FeedBack Chatbot  </h1><p>Efficient Query</p></div>",
                unsafe_allow_html=True)
    df = pd.read_csv("C:/Users/sakth/Downloads/customer_feedback_sentiment (1).csv", encoding='latin-1')
    def assign(Rating):
        if Rating == 1 or Rating == 2:
            return "Negative"
        elif Rating == 3:
            return "Neutral"
        else:
            return "Positive"
    df['Sentimentlabel'] = df['Rating'].apply(assign)
    df = df[["Review_Text", "Sentimentlabel"]]
    stop_words = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()
    def clean(text):
        if text is None:
            return ''
        text = re.sub(r'https\S+', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = word_tokenize(text)
        text = [lemma.lemmatize(word, pos='v') for word in text if word not in stop_words and len(word) > 2]
        return ' '.join(text)
    df['customerfeedback'] = df['Review_Text'].apply(clean)
    df = df.drop('Review_Text', axis=1)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['customerfeedback'])
    y = df['Sentimentlabel']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    def predict_sentiment(text):
        processed_text = clean(text)
        vectorized_text = vectorizer.transform([processed_text])
        return model.predict(vectorized_text)[0]
    st.image("C:/Users/sakth/Downloads\chatbot1.png")
    st.write(df.head(10))
    feedback = st.text_input("Enter your feedback:")
    if st.button("Submit"):
        sentiment = predict_sentiment(feedback)
        st.write(f'Accuracy is  {accuracy * 100:.2f}%')
        st.write(f"Sentiment Prediction: **{sentiment}**")
    wordcloud = WordCloud(width=800, height=300, background_color='white').generate(''.join(df['customerfeedback']))
    def word_cloud():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    word_cloud()
elif selection=='Insurance Risk_Score':
    st.markdown(
        f"""
                 <style>
                    .stApp {{
                     background-color:#E6E6FA

                 }}
                    .main-content {{
                         padding: 20px;
                         border-radius: 10px;
                         background-color: 	rgb(245, 222, 179); /* Add slight transparency */
                         color: black;
                         text-align: center;
                     }}
                 </style>
                 """,
        unsafe_allow_html=True)
    st.markdown("<div class='main-content'><h1>Risk Score </h1><p> Insurance Policy </p></div>",
                unsafe_allow_html=True)
    st.image("C:/Users/sakth/Downloads/2204292.webp")
    col1,col2=st.columns(2)
    with col1:
        Age=st.number_input("Age")
        Annual_Income=st.number_input("Annual_Income")
        Claim_History=st.number_input("Claim_History")
        Claim_Amount=st.number_input("Claim_Amount")
        Fraud_Label=st.number_input("Fraud_Label")
        Premium_Amount=st.number_input("Premium_Amount")
    with col2:
        Policy_Type_Health=st.number_input("Policy_Type_Health")
        Policy_Type_Life=st.number_input("Policy_Type_Life")
        Policy_Type_Property=st.number_input("Policy_Type_Property")
        Gender_Female=st.number_input("Gender_Female")
        Gender_Male=st.number_input("Gender_Male")
        Gender_Other=st.number_input("Gender_Other")
    inputdata3=np.array([Age,Annual_Income,Claim_History,Claim_Amount,Fraud_Label,Premium_Amount,Policy_Type_Health	,Policy_Type_Life,
                         Policy_Type_Property,Gender_Female,Gender_Male	,Gender_Other]).reshape(1,-1)
    if st.button("PREDICT"):
        prediction = model2.predict(inputdata3)
        st.success(*prediction)
elif selection=='Insurance Claim_Amount':
    st.markdown(
        f"""
                     <style>
                        .stApp {{
                         background-color:#FF6347

                     }}
                        .main-content {{
                             padding: 20px;
                             border-radius: 10px;
                             background-color: 	rgb(250, 128, 114); /* Add slight transparency */
                             color: black;
                             text-align: center;
                         }}
                     </style>
                     """,
        unsafe_allow_html=True)
    st.markdown("<div class='main-content'><h1> Claim Amount  </h1><p>  </p></div>",
                unsafe_allow_html=True)
    st.image("C:/Users\sakth\Downloads\9-99913_cash-png.png")
    col1,col2=st.columns(2)
    with col1:
        Age=st.number_input("Age")
        Annual_Income=st.number_input("Annual_Income")
        Claim_History=st.number_input("Claim_History")
        Risk_Score=st.number_input("Risk_Score")
        Fraud_Label=st.number_input("Fraud_Label")
        Premium_Amount=st.number_input("Premium_Amount")
    with col2:
        Policy_Type_Health=st.number_input("Policy_Type_Health")
        Policy_Type_Life=st.number_input("Policy_Type_Life")
        Policy_Type_Property=st.number_input("Policy_Type_Property")
        Gender_Female=st.number_input("Gender_Female")
        Gender_Male=st.number_input("Gender_Male")
        Gender_Other=st.number_input("Gender_Other")
    inputdata5=np.array([Age,Annual_Income,Claim_History,Risk_Score,Fraud_Label,Premium_Amount,Policy_Type_Health,Policy_Type_Life,
                        Policy_Type_Property,Gender_Female,Gender_Male,Gender_Other]).reshape(1,-1)
    if st.button("PREDICT"):
        prediction=model3.predict(inputdata5)
        st.write(*prediction)












