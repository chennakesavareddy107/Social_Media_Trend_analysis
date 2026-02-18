import streamlit as st
from newspaper import Article
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from collections import Counter

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text, article.title
    except Exception as e:
        return None, f"Error: {e}"

def get_trends(text):
    # Basic keyword extraction as a "trend" indicator
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text.lower())
    # Filter alpha-only and non-stopwords
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 3]
    return Counter(filtered_words).most_common(10)

# --- Streamlit UI ---
st.set_page_config(page_title="Social Trend Analyzer", layout="wide")

st.title("📈 Social Media Trend & Sentiment Analysis")
st.markdown("Analyze the 'buzz' from any social media link or news article.")

url_input = st.text_input("Paste Social Media/Article URL here:", placeholder="https://...")

if url_input:
    with st.spinner("Extracting and analyzing content..."):
        text, title = extract_text_from_url(url_input)
        
        if text:
            st.subheader(f"Analyzing: {title}")
            
            # Layout Columns
            col1, col2 = st.columns(2)
            
            with col1:
                # 1. Sentiment Analysis
                analysis = TextBlob(text)
                polarity = analysis.sentiment.polarity
                
                st.write("### 🎭 Sentiment Score")
                if polarity > 0:
                    st.success(f"Positive ({round(polarity, 2)})")
                elif polarity < 0:
                    st.error(f"Negative ({round(polarity, 2)})")
                else:
                    st.info("Neutral")
                
                # 2. Key Trends (Keywords)
                st.write("### 🔥 Top Keywords (Trends)")
                trends = get_trends(text)
                for word, count in trends:
                    st.write(f"- **{word.capitalize()}**: Mentioned {count} times")

            with col2:
                # 3. Word Cloud
                st.write("### ☁️ Trend Visualizer")
                wc = WordCloud(background_color="white", width=800, height=400).generate(text)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
                
            st.divider()
            st.write("### 📝 Extracted Content Snippet")
            st.write(text[:1000] + "...")
        else:
            st.error("Could not extract text. Ensure the URL is public and contains readable text.")
