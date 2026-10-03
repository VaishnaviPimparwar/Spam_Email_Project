import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0F172A, #1E293B);
    color: white;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #38BDF8;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Text Area */
div[data-testid="stTextArea"] textarea {
    background-color: #334155;
    color: white;
    border-radius: 12px;
    border: 2px solid #38BDF8;
    font-size: 16px;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg,#38BDF8,#06B6D4);
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(90deg,#0EA5E9,#0891B2);
}

/* Result Card */
.result-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

/* Footer */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 12px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.markdown(
    '<div class="main-title">📧 Spam Email Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Based Email Classifier using Naive Bayes</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD DATASET ---------------- #
try:
    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    df['label'] = df['label'].map({
        'ham': 0,
        'spam': 1
    })

    x = df['message']
    y = df['label']

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    cv = CountVectorizer()

    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train, y_train)

except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# ---------------- INPUT SECTION ---------------- #
col1, col2, col3 = st.columns([1, 4, 1])

with col2:

    st.subheader("✉️ Enter Email Message")

    input_email = st.text_area(
        "",
        height=220,
        placeholder="Paste or type your email content here..."
    )

    if st.button("🔍 Detect Email"):

        if input_email.strip() == "":
            st.warning("Please enter an email message.")
        else:

            input_data = cv.transform([input_email])

            prediction = model.predict(input_data)

            probability = model.predict_proba(input_data)

            confidence = max(probability[0]) * 100

            if prediction[0] == 1:
                st.markdown(
                    """
                    <div class="result-card"
                    style="background:#7F1D1D;color:#FCA5A5;">
                    🚨 SPAM EMAIL DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-card"
                    style="background:#14532D;color:#86EFAC;">
                    ✅ NORMAL EMAIL
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("### 📊 Prediction Confidence")
            st.progress(int(confidence))
            st.success(f"{confidence:.2f}% Confidence")

# ---------------- FOOTER ---------------- #
st.markdown(
    """
    <div class="footer">
        💡 Spam Email Detection Project
    </div>
    """,
    unsafe_allow_html=True
)