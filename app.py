import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Clean & Find - AI Data Tool", layout="wide")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def clean_dataset(df):
    report = {}
    report["initial_shape"] = f"{df.shape[0]} x {df.shape[1]}"
    report["missing_values_before"] = df.isnull().sum().to_dict()
    df = df.drop_duplicates()
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    report["final_shape"] = f"{df.shape[0]} x {df.shape[1]}"
    report["missing_values_after"] = df.isnull().sum().to_dict()
    return df, report

st.title("Clean & Find")
st.markdown('<h5 style="text-align:center; color:gray;">AI-Powered Data Cleaning & Semantic Search</h5>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df_cleaned, report = clean_dataset(df)
    st.session_state["df_cleaned"] = df_cleaned

    st.subheader("Cleaning Report")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before Cleaning**")
        st.write("Shape:", report["initial_shape"])
        st.write("Missing Values:", report["missing_values_before"])
    with col2:
        st.markdown("**After Cleaning**")
        st.write("Shape:", report["final_shape"])
        st.write("Missing Values:", report["missing_values_after"])

    # Bar chart for missing values
    st.subheader("Missing Values Chart")
    mv_before = pd.Series(report["missing_values_before"])
    mv_after = pd.Series(report["missing_values_after"])
    mv_df = pd.DataFrame({"Before Cleaning": mv_before, "After Cleaning": mv_after})
    st.bar_chart(mv_df)

    # Download cleaned dataset
    output = BytesIO()
    df_cleaned.to_csv(output, index=False)
    output.seek(0)
    st.download_button(
        label="Download Cleaned Dataset",
        data=output,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    # Semantic Search
    st.subheader("Semantic Search")
    search_column = st.selectbox("Select Search Column", df_cleaned.columns)
    texts = df_cleaned[search_column].astype(str).tolist()
    embeddings = model.encode(texts)

    query = st.text_input("Enter your search query")
    if query:
        query_emb = model.encode([query])
        scores = cosine_similarity(query_emb, embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:5]
        results = df_cleaned.iloc[top_idx][search_column].astype(str).tolist()
        st.markdown("**Top 5 results:**")
        for i, res in enumerate(results, 1):
            st.write(f"{i}. {res}")
else:
    st.info("Please upload a CSV or Excel file to begin.")