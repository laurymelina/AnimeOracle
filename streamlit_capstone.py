import streamlit as st
import pandas as pd
import boto3
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600)
def load_s3_data_chunked(bucket_name, file_name, usecols=None):
    """Load large S3 files in chunks"""
    try:
        s3 = boto3.client('s3')
        chunks = []
        
        # Read in chunks with only necessary columns
        for chunk in pd.read_csv(
            StringIO(
                s3.get_object(Bucket=bucket_name, Key=file_name)["Body"].read().decode("utf-8")
            ),
            chunksize=10000,  # Adjust chunk size based on memory
            usecols=usecols
        ):
            # Optimize datatypes
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = chunk[col].astype('float32')
            for col in chunk.select_dtypes(include=['int64']).columns:
                chunk[col] = chunk[col].astype('int32')
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        logger.error(f"Error loading {file_name} from S3: {e}")
        st.error(f"Failed to load {file_name}. Please try again later.")
        raise e

@st.cache_data(ttl=3600)
def load_local_data(file_name):
    """Load local files"""
    try:
        df = pd.read_csv(file_name)
        return df
    except Exception as e:
        logger.error(f"Error loading {file_name}: {e}")
        st.error(f"Failed to load {file_name}")
        raise e

def load_initial_data():
    """Load only essential data for initial app startup"""
    try:
        with st.spinner('Loading initial data...'):
            # Load small local files first
            genre_list = load_local_data("genre_list.csv")
            studio_list = load_local_data("studio_list.csv")
            title_list = load_local_data("title_list.csv")
            status_list = load_local_data("status_list.csv")
            episodes_list = load_local_data("episodes_list.csv")
            started_list = load_local_data("started_list.csv")
            
            # Load only essential columns from anime_final
            essential_cols = ['anime_id', 'title', 'score', 'score_count', 'anime_url', 'main_pic']
            bucket_name = "animeoracle"
            anime_final = load_s3_data_chunked(bucket_name, "final_df.csv", usecols=essential_cols)
            
            return {
                'genre_list': genre_list,
                'studio_list': studio_list,
                'title_list': title_list,
                'status_list': status_list,
                'episodes_list': episodes_list,
                'started_list': started_list,
                'anime_final': anime_final
            }
    except Exception as e:
        logger.error(f"Error in load_initial_data: {e}")
        st.error("Failed to load initial data. Please refresh the page.")
        raise e

@st.cache_data(ttl=3600)
def load_recommendation_data(selected_titles):
    """Load similarity data only when needed for recommendations"""
    try:
        bucket_name = "animeoracle"
        
        # Load only the columns we need from similarity matrices
        tfidf_cols = ['title'] + selected_titles
        tags_cols = ['title'] + selected_titles
        
        tfidf = load_s3_data_chunked(bucket_name, "tfidf_similarity_df.csv", usecols=tfidf_cols)
        tags = load_s3_data_chunked(bucket_name, "tag_similarities_df.csv", usecols=tags_cols)
        
        return tfidf, tags
    except Exception as e:
        logger.error(f"Error loading recommendation data: {e}")
        st.error("Failed to load recommendation data. Please try again.")
        raise e

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Main app initialization
try:
    if st.session_state.data is None:
        st.session_state.data = load_initial_data()
    
    # Access data from session state
    anime_final = st.session_state.data['anime_final']
    genre_list = st.session_state.data['genre_list']
    tfidf = st.session_state.data['tfidf']
    tags = st.session_state.data['tags']
    studio_list = st.session_state.data['studio_list']
    title_list = st.session_state.data['title_list']
    status_list = st.session_state.data['status_list']
    episodes_list = st.session_state.data['episodes_list']
    started_list = st.session_state.data['started_list']
    
    # Rest of your app code...
    

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

    # Initialize session state attributes
    if 'random_anime' not in st.session_state:
        st.session_state.random_anime = None
    if 'selected_anime' not in st.session_state:
        st.session_state.selected_anime = []

    # Function to get random anime
    def get_random_anime():
        random_anime_titles = list(title_list['title'])  # Using the title column
        filtered_anime_final = anime_final.loc[anime_final['title'].isin(random_anime_titles)]
        st.session_state.random_anime = filtered_anime_final.sample(9, replace=False)
        st.experimental_rerun()  # Force Streamlit to rerun the script


    # Select Your Favorite Anime
    st.markdown("<h2 style='text-align: left; color: #2e86c1;'>Select Your Favorite Anime:</h2>", unsafe_allow_html=True)

    def on_change():
        if st.session_state.title_selectbox:
            if st.session_state.title_selectbox not in st.session_state.selected_anime:
                st.session_state.selected_anime.append(st.session_state.title_selectbox)
                st.success(f"Added '{st.session_state.title_selectbox}' to your selection.")
            # Reset the selectbox
            st.session_state.title_selectbox = ""

    # Initialize session state variables if they don't exist
    if 'selected_anime' not in st.session_state:
        st.session_state.selected_anime = []

    # Dropdown to select a title from the full list
    st.markdown("""
        <div style='margin-bottom: -40px;'>
            <h4 style='text-align: left; color: #2e86c1;'>Search For Title:</h4>
        </div>
    """, unsafe_allow_html=True)

    selected_title = st.selectbox("", 
                                [""] + sorted(anime_final['title'].tolist()), 
                                key='title_selectbox',
                                on_change=on_change)



    # Load random anime once, and keep it in session state
    if st.session_state.random_anime is None:
        get_random_anime()


    st.markdown("""
        <style>
        .anime-selection {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .anime-selection .stCheckbox {
            flex: 0 0 auto;
            margin-right: 10px;
        }
        .anime-selection .anime-title {
            flex: 1 1 auto;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        /* Adjust checkbox size and alignment */
        .anime-selection .stCheckbox > label > div[role="checkbox"] {
            width: 20px !important;
            height: 20px !important;
        }
        .anime-selection .stCheckbox > label {
            display: flex !important;
            align-items: center !important;
        }
        .anime-title a {
            color: #3498db;  /* Set the link color to match your app's theme */
            text-decoration: none;  /* Remove the default underline */
        }

        .anime-title a:hover {
            text-decoration: underline;  /* Add underline on hover */
        }
        </style>
        """, unsafe_allow_html=True)

    # Section Header
    st.markdown("<h4 style='text-align: left; color: #3498db;'>Or Select From Most Popular:</h4>", unsafe_allow_html=True)

    # Display random anime and selection checkboxes
    for i in range(0, 9, 3):  # This will create 3 rows
        cols = st.columns(3)
        for j in range(3):  # This will create 3 columns in each row
            if i + j < len(st.session_state.random_anime):
                anime = st.session_state.random_anime.iloc[i + j]
                with cols[j]:
                    st.image(anime['main_pic'], use_column_width=True)
                    
                    # Create a container for checkbox and title
                    col1, col2 = st.columns([0.1, 0.9])
                    with col1:
                        checked = st.checkbox('', value=anime['title'] in st.session_state.selected_anime, key=f"checkbox_{anime['anime_id']}", label_visibility="collapsed")
                    with col2:
                        st.markdown(f'<div class="anime-title"><a href="{anime["anime_url"]}" target="_blank">{anime["title"]}</a></div>', unsafe_allow_html=True)
                    
                    # Handle checkbox state
                    if checked:
                        if anime['title'] not in st.session_state.selected_anime:
                            st.session_state.selected_anime.append(anime['title'])
                    else:
                        if anime['title'] in st.session_state.selected_anime:
                            st.session_state.selected_anime.remove(anime['title'])

    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');

        div.stButton > button {
            background-color: #FF5733; /* Orange background */
            color: white; /* White text */
            font-family: 'Roboto', sans-serif; /* Use imported font */
            font-size: 18px;
            padding: 5px 12px;
            border-radius: 10px;
            border: 2px solid #FF5733;
            cursor: pointer;
            transition: background-color 0.3s, border-color 0.3s;
        }

        div.stButton > button:hover {
            background-color: #cb4335; /* Darker shade on hover */
            border-color: #cb4335; /* Change border on hover */
            color: white !important; /* Ensure text stays white on hover */
        }
        </style>
        """, unsafe_allow_html=True)

    # Add extra space
    st.markdown("<br>", unsafe_allow_html=True)  # Adds two line breaks

    # Refresh button for new anime selections
    if st.button("Refresh Random Anime Selection"):
        get_random_anime()


    # Add extra space
    st.markdown("<br>", unsafe_allow_html=True)  # Adds two line breaks

    # Display currently selected anime
    if st.session_state.selected_anime:
        #st.write("**Currently selected anime:**")
        st.markdown("<h5 style='text-align: left; color: #2e86c1;'>Currently Selected Anime:</h5>", unsafe_allow_html=True)
        for idx, anime in enumerate(st.session_state.selected_anime):
            col1, col2 = st.columns([0.1, 1.5])
            if col1.button("x", key=f"remove_{idx}"):
                st.session_state.selected_anime.remove(anime)
                st.experimental_rerun()
            col2.write(f"**{anime}**")


    st.markdown("""
        <h4 style='text-align: left; color: #2e86c1; font-size: 18px;'> 
            <span title="Give a recommendation based on the anime description being more similar">Synopsis Weight:*</span>
        </h4>
    """, unsafe_allow_html=True)

    tfidf_weight = st.slider("", min_value=1, max_value=5, value=1, step=1, key='tfidf_weight_slider')

    # Slider for Tags Weight with tooltip
    st.markdown("""
        <h4 style='text-align: left; color: #2e86c1; font-size: 18px;'>
            <span title="Give a recommendation based on the anime tags being more similar like genre, studio, year started, etc.">Tags Weight:*</span>
        </h4>
    """, unsafe_allow_html=True)

    tag_weight = st.slider("", min_value=1, max_value=5, value=1, step=1, key='tfidf_tag_slider')

    # Optional filters section
    st.sidebar.markdown("**Filters**")

    # Status filter
    status_display_to_column = dict(zip(status_list.iloc[:, 0], status_list.iloc[:, 1]))
    selected_status_display = st.sidebar.multiselect("Airing Status:", options=status_display_to_column.keys())
    selected_status = [status_display_to_column[g] for g in selected_status_display]

    # Genres filter
    genre_display_to_column = dict(zip(genre_list.iloc[:, 0], genre_list.iloc[:, 1]))
    selected_genres_display = st.sidebar.multiselect("Genres:", options=genre_display_to_column.keys())
    selected_genres = [genre_display_to_column[g] for g in selected_genres_display]

    # Episodes filter
    episodes_display_to_column = dict(zip(episodes_list.iloc[:, 0], episodes_list.iloc[:, 1]))
    selected_episodes_display = st.sidebar.multiselect("Number of Episodes:", options=episodes_display_to_column.keys())
    selected_episodes = [episodes_display_to_column[g] for g in selected_episodes_display]

    # Start Year filter
    started_display_to_column = dict(zip(started_list.iloc[:, 0], started_list.iloc[:, 1]))
    selected_started_display = st.sidebar.multiselect("Year Anime Started:", options=started_display_to_column.keys())
    selected_started = [started_display_to_column[g] for g in selected_started_display]

    # Studios filter
    studio_display_to_column = dict(zip(studio_list.iloc[:, 0], studio_list.iloc[:, 1]))
    selected_studio_display = st.sidebar.multiselect("Studios:", options=studio_display_to_column.keys())
    selected_studio = [studio_display_to_column[g] for g in selected_studio_display]

    # Rating slider
    min_rating, max_rating = st.sidebar.slider("Rating:", min_value=1, max_value=10, value=(1, 10))

    # Score Count slider
    score_count = st.sidebar.slider("Number of Users Scoring:", min_value=0, max_value=50000, value=0, step=5000)


    def anime_recommender(df, titles, status_columns, genre_columns, episode_columns, started_columns, studio_columns, min_rating=0, max_rating=10, tfidf_weight=1, tag_weight=1, score_count=1):
        # Step 1: Set up a new DataFrame with necessary columns
        columns_to_copy = ['anime_id', 'title', 'score', 'score_count', 'anime_url', 'main_pic']
        
        # Dynamically pull only the related_ columns for the selected anime based on their anime_id
        prefixed_columns = []
        for title in titles:
            # Get the anime_id corresponding to the title
            anime_id = df.loc[df['title'] == title, 'anime_id'].values[0]
            related_column = f'related_{anime_id}' 
            prefixed_columns.append(related_column)

        columns_to_copy.extend(prefixed_columns)
        new_df = df[columns_to_copy]

        # Step 2: Add TF-IDF and tag scores for each title 
        for title in titles:
            tfidf_col_name = f'tfidf_{title}'
            tag_col_name = f'tag_{title}'
            new_df[tfidf_col_name] = tfidf[title] * tfidf_weight
            new_df[tag_col_name] = tags[title] * tag_weight

        # Step 3: Apply any filters selected for the different tags

        # Status filter
        if status_columns:
            new_df['total_status'] = df[status_columns].sum(axis=1)
        else:
            new_df['total_status'] = 0.001

        new_df = new_df.loc[new_df['total_status'] > 0]

        # Genre filter
        if genre_columns:
            new_df['total_genres'] = df[genre_columns].sum(axis=1)
        else:
            new_df['total_genres'] = 0.001

        new_df = new_df.loc[new_df['total_genres'] > 0]

        # Episode filter
        if episode_columns:
            new_df['total_episodes'] = df[episode_columns].sum(axis=1)
        else:
            new_df['total_episodes'] = 0.001

        new_df = new_df.loc[new_df['total_episodes'] > 0]

        # Start Date filter
        if started_columns:
            new_df['total_start_date'] = df[started_columns].sum(axis=1)
        else:
            new_df['total_start_date'] = 0.001

        new_df = new_df.loc[new_df['total_start_date'] > 0]

        # Studio filter
        if studio_columns:
            new_df['total_studio'] = df[studio_columns].sum(axis=1)
        else:
            new_df['total_studio'] = 0.001

        new_df = new_df.loc[new_df['total_studio'] > 0]

        # Rating filter
        new_df = new_df.loc[(new_df['score'] >= min_rating) & (new_df['score'] <= max_rating)]

        # Score count filter
        new_df = new_df.loc[(new_df['score_count'] >= score_count)]

        # Step 4: Filter out rows where anime_id is in related_anime_ids
        # Initialize an empty set to track anime_ids related to the titles
        related_anime_ids = set()
        
        for title in titles:
            # Find anime that are related to the current title using related_ columns
            related_columns = [col for col in new_df.columns if col.startswith('related_')]

            for col in related_columns:
                # Check if this column relates to the current anime's title
                if df['anime_id'][df['title'] == title].empty:
                    continue  # Skip if there's no anime_id for the title

                anime_id = df['anime_id'][df['title'] == title].values[0]

                # If the related column refers to this title's anime_id and has 1s, collect the related anime_ids
                if col == f'related_{anime_id}':
                    related_anime_ids.update(new_df.loc[new_df[col] == 1, 'anime_id'])

        new_df = new_df[~new_df['anime_id'].isin(related_anime_ids)]

        # Create a new 'total_title' column to add up the tfidf and tag columns created and then divide by the total number of anime selected
        tfidf_tag_columns = [col for col in new_df.columns if col.startswith('tfidf_') or col.startswith('tag_')]
        new_df['total_title'] = new_df[tfidf_tag_columns].sum(axis=1) / (len(titles))
        new_df['final_total'] = new_df['total_title'] + new_df['total_genres'] + new_df['total_studio']

        sorted_df = new_df.sort_values(by='final_total', ascending=False).head(50)

        return sorted_df

    # Add extra space
    st.markdown("<br>", unsafe_allow_html=True)  # Adds two line breaks

    # Summon button
    if st.button("**Summon!**"):
        if not st.session_state.selected_anime:
            st.error("Please select at least one anime.")
        else:
            # Load similarity data only when needed
            with st.spinner('Loading recommendation data...'):
                tfidf, tags = load_recommendation_data(st.session_state.selected_anime)
                
            recommendations = anime_recommender(
                anime_final, 
                st.session_state.selected_anime,
                selected_status,
                selected_genres,
                selected_episodes,
                selected_started,
                selected_studio,
                min_rating,
                max_rating,
                tfidf_weight=tfidf_weight,
                tag_weight=tag_weight,
                score_count=score_count,
                tfidf_data=tfidf,
                tags_data=tags
            )
            # Randomly select 6 from the top 50 recommendations
            top_recommendations = recommendations.head(50)
            final_recommendations = top_recommendations.sample(6)

            # Display recommendations
            st.markdown("### Recommendations:")
            for i in range(0, 6, 3):  # This will create 2 rows
                cols = st.columns(3)
                for j in range(3):  # This will create 3 columns in each row
                    if i + j < len(final_recommendations):
                        anime = final_recommendations.iloc[i + j]
                        with cols[j]:
                            st.image(anime['main_pic'], width=150)
                            st.markdown(f"[{anime['title']}]({anime['anime_url']})")
                            st.markdown(f"**Score:** {anime['score']:.2f}")


except Exception as e:
    logger.error(f"Application initialization error: {e}")
    st.error("Failed to initialize the application. Please refresh the page.")
