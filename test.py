import streamlit as st
import pandas as pd
import random

# Load data with caching
@st.cache_data
def load_data():
    anime_final = pd.read_csv("final_df.csv")
    tfidf = pd.read_csv("tfidf_similarity_df.csv")
    tags = pd.read_csv("tag_similarities_df.csv")
    genre_list = pd.read_csv("genre_list.csv")
    studio_list = pd.read_csv("studio_list.csv")
    title_list = pd.read_csv("title_list.csv")
    status_list = pd.read_csv("status_list.csv")
    episodes_list = pd.read_csv("episodes_list.csv")
    started_list = pd.read_csv("started_list.csv")
    return anime_final, tfidf, tags, genre_list, studio_list, title_list, status_list,episodes_list, started_list

anime_final, tfidf, tags, genre_list, studio_list, title_list, status_list, episodes_list, started_list = load_data()

# CSS for background image with transparency
st.markdown(
    """
    <style>
    .top-section {
        background-image: url("https://beebom.com/wp-content/uploads/2023/06/Anime.jpg?quality=75&strip=all");
        background-size: cover;
        background-position: center;
        padding: 50px 0;  /* Adjust the padding to control the height of the image section */
        position: relative;  /* Ensure child elements are positioned relative to this container */
    }

    .top-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);  /* Add a black overlay with 50% transparency */
        z-index: 1;
    }

    .top-section h1 {
        position: relative;  /* Position the title relative to the section */
        z-index: 2;
        text-align: center;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Content of the top section with the background image
st.markdown(
    """
    <div class="top-section">
        <h1>Anime Oracle: Predicting Your Next Favorite Anime</h1>
    </div>
    """, unsafe_allow_html=True
)