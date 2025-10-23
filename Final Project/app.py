import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
import gradio as gr
import requests
from io import BytesIO
from PIL import Image
import base64

# ----------------------------
# TMDb API settings
# ----------------------------
TMDB_API_KEY = "69349e2143c88bfd29e023e891e50134"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_movie_poster_base64(title):
    """Fetch poster from TMDb and return as base64 string for HTML"""
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(search_url, params=params).json()
    results = response.get("results")
    if results:
        poster_path = results[0].get("poster_path")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}"  # smaller width for display
            try:
                img_response = requests.get(poster_url)
                img = Image.open(BytesIO(img_response.content))
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_str}"
            except:
                return ""
    return ""

# ----------------------------
# Load data and vectorizer
# ----------------------------
df = pd.read_csv("data/movies_cleaned.csv")
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def get_movie_recommendations_html(movie_name, top_n=5):
    if movie_name not in df['title'].values:
        return "<p>No recommendations found.</p>"

    # TF-IDF for text_features
    X_text = tfidf.transform(df['text_features'])

    # Multi-hot encoding
    multi_hot_cols = ['genres_list','keywords_list','cast_list','crew_list']
    X_multi = []
    for col in multi_hot_cols:
        if col in df.columns:
            mlb = MultiLabelBinarizer(sparse_output=True)
            X_col = mlb.fit_transform(df[col])
            X_multi.append(X_col)

    X = hstack([X_text] + X_multi) if X_multi else X_text

    # Compute similarity
    idx = df.index[df['title'] == movie_name][0]
    sim_scores = cosine_similarity(X[idx], X)[0]

    df['similarity'] = sim_scores
    recommendations = df.loc[df.index != idx].sort_values('similarity', ascending=False).head(top_n)

    # Build HTML cards
    html_output = ""
    for _, row in recommendations.iterrows():
        poster = get_movie_poster_base64(row['title'])
        html_output += f"""
        <div style="display:flex; margin-bottom:20px; align-items:flex-start;">
            <img src="{poster}" width="120" style="margin-right:10px"/>
            <div>
                <h3>{row['title']}</h3>
                <p><b>Similarity:</b> {row['similarity']:.3f}</p>
                <p>{row['text_features']}</p>
            </div>
        </div>
        <hr>
        """
    return html_output

# ----------------------------
# Gradio interface
# ----------------------------
iface = gr.Interface(
    fn=get_movie_recommendations_html,
    inputs=gr.Textbox(label="Enter Movie Title"),
    outputs=gr.HTML(),  # HTML output to render rows
    title="Movie Recommendation System"
)

iface.launch()
#https://huggingface.co/spaces/Maryyyyyyyyyyyyyyyyy/movie_recom