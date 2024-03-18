import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import sqlite3
@st.cache_resource(allow_output_mutation=True)
def get_connection():
    return sqlite3.connect("email_db.sqlite")
# Load data
data = pd.read_csv("spam.csv")
data['spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
data = data.drop('Category', axis=1)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://images.unsplash.com/photo-1532024802178-20dbc87a312a?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1470&q=80");
background-size: 100%;
display: flex;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Split data
x = data['Message']
y = data['spam']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train model
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(x_train, y_train)

# Streamlit UI
st.title("Email Spam Detector")
st.markdown("Please enter the mail below to check if it is ham or spam:")
mail = st.text_input("Enter your mail here:")
st.text("Your result will display!!")

# Check mail
if st.button("Check"):
    res = clf.predict([mail])[0]
    if res == 1:
        st.text("Oops! The mail is a spam!")
    else:
        st.text("Hurray!! It is not a spam mail!")

    try:
        conn = get_connection()
        conn.execute('INSERT INTO Email_checkers (Mail, Result) VALUES (?, ?)', (mail, "Spam" if res == 1 else "Ham"))
        conn.commit()
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")

