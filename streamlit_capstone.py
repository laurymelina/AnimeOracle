import streamlit as st
import pandas as pd
import boto3
from io import StringIO
import logging
import gc  # For garbage collection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Increase the default timeout for data loading
st.set_page_config(layout="wide")

@st.cache_data(ttl=3600)
def load_s3_data_chunked(bucket_name, file_name, usecols=None, chunksize=5000):
    """Load large S3 files in chunks with optimized memory usage"""
    try:
        s3 = boto3.client('s3')
        chunks = []
        
        # Get the S3 object only once
        s3_object = s3.get_object(Bucket=bucket_name, Key=file_name)["Body"].read().decode("utf-8")
        
        # Process in smaller chunks
        for chunk in pd.read_csv(
            StringIO(s3_object),
            chunksize=chunksize,
            usecols=usecols,
            dtype_backend='numpy_nullable'  # More memory efficient
        ):
            # Optimize datatypes
            for col in chunk.select_dtypes(include=['float64']).columns:
                chunk[col] = chunk[col].astype('float32')
            for col in chunk.select_dtypes(include=['int64']).columns:
                chunk[col] = chunk[col].astype('int32')
            chunks.append(chunk)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        # Concatenate all chunks
        result = pd.concat(chunks, ignore_index=True)
        
        # Clear the chunks list to free memory
        chunks.clear()
        del s3_object
        gc.collect()
        
        return result
    except Exception as e:
        logger.error(f"Error loading {file_name} from S3: {e}")
        st.error(f"Failed to load {file_name}. Please try again later.")
        raise e

@st.cache_data(ttl=3600)
def load_recommendation_data(selected_titles):
    """Load and optimize similarity data"""
    try:
        bucket_name = "animeoracle"
        
        # Load only the columns we need from similarity matrices
        tfidf_cols = ['title'] + selected_titles
        tags_cols = ['title'] + selected_titles
        
        with st.spinner('Loading TF-IDF similarities...'):
            tfidf = load_s3_data_chunked(bucket_name, "tfidf_similarity_df.csv", 
                                       usecols=tfidf_cols, chunksize=5000)
        
        with st.spinner('Loading tag similarities...'):
            tags = load_s3_data_chunked(bucket_name, "tag_similarities_df.csv", 
                                      usecols=tags_cols, chunksize=5000)
        
        return tfidf, tags
    except Exception as e:
        logger.error(f"Error loading recommendation data: {e}")
        st.error("Failed to load recommendation data. Please try again.")
        raise e

def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def anime_recommender(df, titles, status_columns, genre_columns, episode_columns, 
                     started_columns, studio_columns, min_rating, max_rating, 
                     tfidf_weight, tag_weight, score_count, tfidf_data, tags_data):
    """Optimized recommendation function with progress tracking"""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Set up a new DataFrame with necessary columns
        status_text.text("Preparing data...")
        progress_bar.progress(10)
        
        columns_to_copy = ['anime_id', 'title', 'score', 'score_count', 'anime_url', 'main_pic']
        new_df = df[columns_to_copy].copy()
        new_df = optimize_dataframe(new_df)
        
        progress_bar.progress(20)
        
        # Step 2: Calculate similarity scores in batches
        status_text.text("Calculating similarities...")
        total_score = pd.Series(0, index=new_df.index, dtype='float32')
        
        for idx, title in enumerate(titles):
            # Update progress
            progress = 30 + (idx * 10 // len(titles))
            progress_bar.progress(progress)
            
            # Calculate combined score for this title
            tfidf_score = tfidf_data[title] * tfidf_weight
            tag_score = tags_data[title] * tag_weight
            total_score += (tfidf_score + tag_score)
        
        new_df['similarity_score'] = total_score / len(titles)
        progress_bar.progress(60)
        
        # Step 3: Apply filters
        status_text.text("Applying filters...")
        
        # Filter by rating and score count first (these are simple comparisons)
        mask = (
            (new_df['score'] >= min_rating) & 
            (new_df['score'] <= max_rating) & 
            (new_df['score_count'] >= score_count)
        )
        new_df = new_df[mask]
        
        progress_bar.progress(80)
        
        # Sort by similarity score
        new_df = new_df.sort_values('similarity_score', ascending=False)
        
        # Get top 50 recommendations
        recommendations = new_df.head(50)
        
        progress_bar.progress(100)
        status_text.text("Done!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error in recommendation process: {e}")
        st.error("An error occurred while generating recommendations. Please try again.")
        raise e

# When the Summon button is clicked
if st.button("**Summon!**"):
    if not st.session_state.selected_anime:
        st.error("Please select at least one anime.")
    else:
        try:
            # Load similarity data
            with st.spinner('Loading recommendation data...'):
                tfidf_data, tags_data = load_recommendation_data(st.session_state.selected_anime)
            
            # Generate recommendations with progress tracking
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
                tfidf_data=tfidf_data,
                tags_data=tags_data
            )
            
            # Sample 6 recommendations from the top 50
            final_recommendations = recommendations.sample(min(6, len(recommendations)))
            
            # Display recommendations
            st.markdown("### Recommendations:")
            cols = st.columns(3)
            for idx, anime in enumerate(final_recommendations.itertuples()):
                with cols[idx % 3]:
                    st.image(anime.main_pic, width=150)
                    st.markdown(f"[{anime.title}]({anime.anime_url})")
                    st.markdown(f"**Score:** {anime.score:.2f}")
            
            # Clear memory
            del tfidf_data, tags_data, recommendations, final_recommendations
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            st.error("Failed to generate recommendations. Please try again.")