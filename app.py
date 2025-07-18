import streamlit as st

# --- CRITICAL: set_page_config MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="భారత్ పల్స్ (BharatPulse) - వార్తలు",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Remaining Imports (after set_page_config) ---
import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article # Make sure newspaper3k and lxml_html_clean are installed
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
import geocoder
import json

# --- Configuration ---
TELUGU_CATEGORIES = {
    "జాతీయ వార్తలు": "National",
    "రాష్ట్ర వార్తలు": "State",
    "క్రైం": "Crime",
    "రాజకీయాలు": "Politics",
    "వ్యాపారం": "Business",
    "క్రీడలు": "Sports",
    "వినోదం": "Entertainment",
    "ఉద్యోగాలు": "Jobs",
    "వాతావరణం": "Weather",
    "వైరల్": "Viral",
}

# --- RSS Feed URLs ---
# These are examples. You might need to verify their current validity and content.
# Some news sites do not offer comprehensive RSS feeds, and scraping might be needed.
RSS_FEEDS = {
    "Sakshi": "https://www.sakshi.com/tags/rss", # This appears to be a general RSS for Sakshi.
    "Eenadu": "https://www.eenadu.net/telugu-news/rss", # This URL from search might be old/unofficial.
    # Way2News does not appear to have a public RSS feed, requiring scraping if you want its content.
    # For now, let's stick to sources that publicly offer RSS.
    # You might consider other major Telugu news sources like ABN Andhra Jyothy, TV9 Telugu if they have RSS.
}

# --- OpenWeatherMap API Configuration (for Realtime Weather) ---
# YOU MUST OBTAIN YOUR OWN API KEY FROM OpenWeatherMap.org and replace the placeholder below.
# A free account gives you access to a key.
OPENWEATHERMAP_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY_HERE" # <--- IMPORTANT: REPLACE THIS WITH YOUR REAL KEY!
OPENWEATHERMAP_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# --- Hugging Face Model Loading (Cached for performance) ---
@st.cache_resource
def load_translation_model():
    """Loads the IndicTrans2 English to Telugu translation model."""
    MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M" # A distilled version (200M parameters)

    try:
        # The 'trust_remote_code=True' is essential for this model.
        # Ensure 'sentencepiece' is installed in your environment (pip install sentencepiece).
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

        # Move model to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")
            st.success("అనువాద మోడల్ విజయవంతంగా లోడ్ చేయబడింది (Translation model loaded successfully) (GPU).")
        else:
            st.success("అనువాద మోడల్ విజయవంతంగా లోడ్ చేయబడింది (Translation model loaded successfully) (CPU).")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading translation model: {e}")
        st.warning("అనువాద ఫీచర్ నిలిపివేయబడుతుంది. దయచేసి 'transformers', 'torch' మరియు 'sentencepiece' సరిగ్గా ఇన్‌స్టాల్ చేయబడి ఉన్నాయని, మీకు తగినంత వనరులు ఉన్నాయని మరియు మోడల్‌ను డౌన్‌లోడ్ చేయడానికి స్థిరమైన ఇంటర్నెట్ కనెక్షన్ ఉందని నిర్ధారించుకోండి. (Translation feature will be disabled. Please ensure 'transformers', 'torch', and 'sentencepiece' are installed correctly, you have sufficient resources, and a stable internet connection to download the model.)")
        st.caption("మీరు 'sentencepiece' గురించి లోపం చూసినట్లయితే, మీ టెర్మినల్‌లో `pip install sentencepiece`ని అమలు చేయండి. (If you see an error about 'sentencepiece', run `pip install sentencepiece` in your terminal.)")
        st.caption("మీరు ప్రమాణీకరణ లోపాలను చూసినట్లయితే, మీరు మీ టెర్మినల్‌లో `huggingface-cli login`ని అమలు చేయాల్సి రావచ్చు. (If you see authentication errors, you might need to run `huggingface-cli login` in your terminal.)")
        return None, None

tokenizer_en_te, model_en_te = load_translation_model()

def translate_en_to_te(text):
    if tokenizer_en_te is None or model_en_te is None:
        return "Translation service unavailable."
    if not text or not text.strip(): # Handle empty or whitespace-only input
        return ""
    try:
        # IndicTrans2 models typically expect language tags in the input for multi-lingual models.
        # <2en> for 'to English', <2te> for 'to Telugu'
        # For EN->TE, the input format for the 'dist' models is `<2te> English_Sentence`
        input_text_with_tags = f"<2te> {text}" 
        
        inputs = tokenizer(input_text_with_tags, return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        translated_tokens = model_en_te.generate(**inputs, max_new_tokens=128, num_beams=5, early_stopping=True)
        translated_text = tokenizer_en_te.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        # Remove any lingering target language tags if present in output
        return translated_text.replace("<2te>", "").strip()
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return "Translation failed."

# --- RSS Feed Fetching and Parsing (Cached for performance) ---
@st.cache_data(ttl=3600) # Cache for 1 hour to reduce API calls to news sites
def get_news_from_rss(url):
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries:
            title = entry.title if hasattr(entry, 'title') else ""
            link = entry.link if hasattr(entry, 'link') else "#"
            summary = entry.summary if hasattr(entry, 'summary') and entry.summary.strip() else ""
            published = entry.published if hasattr(entry, 'published') else "N/A"
            image_url = None

            # Attempt to find image from media_content (common in some RSS feeds)
            if hasattr(entry, 'media_content') and entry.media_content:
                for media in entry.media_content:
                    # Check for actual image type
                    if 'url' in media and media.get('type', '').startswith('image'):
                        image_url = media['url']
                        break
            # Attempt to find image from description HTML (common in others)
            elif hasattr(entry, 'description'):
                soup = BeautifulSoup(entry.description, 'html.parser')
                img_tag = soup.find('img')
                if img_tag and 'src' in img_tag.attrs:
                    image_url = img_tag['src']

            # Fallback to newspaper3k for better image/summary extraction from the article link
            # Only if link is valid and no image/summary found yet
            if link and "http" in link and link != "#" and (not image_url or not summary):
                try:
                    article_parser = Article(link)
                    article_parser.download()
                    article_parser.parse()
                    if not image_url and article_parser.top_image:
                        image_url = article_parser.top_image
                    if not summary and article_parser.text: # Use first part of article text if no summary from RSS
                        summary = article_parser.text[:200] + "..." if len(article_parser.text) > 200 else article_parser.text

                except Exception as e:
                    # print(f"Newspaper3k failed for {link}: {e}") # Uncomment for debugging newspaper issues
                    pass # Silently fail if newspaper3k can't extract

            # Only add articles with a valid title and link
            if title.strip() and link.strip() != "#":
                articles.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "published": published,
                    "image_url": image_url,
                    "translated_title": None # Placeholder for translation
                })
        return articles
    except Exception as e:
        st.error(f"Error fetching news from {url}: {e}")
        return []

# --- Voice Search (Placeholder - requires a separate library/API) ---
def voice_search_widget():
    st.write("### 🎙️ వాయిస్ సెర్చ్ (Voice Search)")
    st.warning("Voice search integration with Whisper requires external setup (e.g., a local Whisper model, an API, or a custom Streamlit component). This is a placeholder.")
    # Example placeholder for voice input (requires user interaction)
    audio_file = st.file_uploader("Upload an audio file (e.g., .wav, .mp3) for transcription:", type=["wav", "mp3"])
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        st.info("Transcribing audio... (This is a mock transcription)")
        # In a real app, you'd send `audio_file` to a Whisper model/API
        st.write("ట్రాన్స్క్రిప్షన్: ఇది వాయిస్ సెర్చ్ కోసం ఒక నకిలీ ట్రాన్స్క్రిప్షన్ (Transcription: This is a mock transcription for voice search)")
        # You would then use the transcribed text for actual search

# --- Location Detection (Placeholder) ---
@st.cache_data(ttl=3600) # Cache location for an hour
def get_user_location():
    """Attempts to get user's public IP based location (city/district)."""
    try:
        g = geocoder.ip('me')
        if g.ok:
            city = g.city if g.city else "Hyderabad"
            state = g.state if g.state else "Telangana"
            country = g.country if g.country else "India"
            return city, state, country
        else:
            return "Hyderabad", "Telangana", "India" # Default to Hyderabad if geocoder fails
    except Exception as e:
        st.warning(f"Could not detect location: {e}. Defaulting to Hyderabad.")
        return "Hyderabad", "Telangana", "India"

# --- Weather Integration ---
@st.cache_data(ttl=600) # Cache weather for 10 minutes
def get_current_weather(city_name, api_key):
    # Check if API key is configured
    if not api_key or api_key == "YOUR_OPENWEATHERMAP_API_KEY_HERE":
        # Do not raise an error here, let the calling function display the warning.
        return None

    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric', # For Celsius
        'lang': 'te' # Try for Telugu, though not all weather APIs support all languages
    }
    try:
        response = requests.get(OPENWEATHERMAP_API_URL, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        weather_data = response.json()

        # Extract relevant info
        main_weather = weather_data['weather'][0]['description'] if weather_data.get('weather') else "N/A"
        temperature = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        humidity = weather_data['main']['humidity']
        wind_speed = weather_data['wind']['speed']

        weather_info = {
            "description": main_weather.capitalize(),
            "temperature": temperature,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed
        }
        return weather_info
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather for {city_name}: {e}. This might be due to an invalid city name or API key issues.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding weather data for {city_name}. Invalid response from API.")
        return None
    except KeyError as e:
        st.error(f"Missing key in weather data for {city_name}: {e}. API response might be incomplete or unexpected.")
        return None

# --- Custom CSS for card-like appearance, horizontal scrolling (no background color) ---
st.markdown("""
<style>
/* Streamlit Tabs Styling */
.stTabs [data-baseweb="tab-list"] button {
    padding: 10px 15px;
    font-size: 16px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 15px; /* Spacing between tabs */
    flex-wrap: nowrap; /* Prevent wrapping */
    overflow-x: auto; /* Enable horizontal scrolling */
    -webkit-overflow-scrolling: touch; /* For smoother scrolling on iOS */
    scrollbar-width: thin; /* Firefox */
    scrollbar-color: #ccc #f1f1f1; /* Firefox */
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 6px; /* Height of the scrollbar */
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background-color: #ccc; /* Color of the scrollbar thumb */
    border-radius: 3px; /* Rounded corners for the thumb */
}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
    background-color: #f1f1f1; /* Color of the scrollbar track */
}

/* News Card Styling */
.news-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 20px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    height: 100%; /* Ensures cards in a row have equal height */
    background-color: #ffffff; /* Explicit white background for cards */
}
.news-card img {
    max-width: 100%;
    height: 200px; /* Fixed height for images */
    object-fit: cover; /* Ensures image covers the area, might crop */
    border-radius: 4px;
    margin-bottom: 10px;
}
.news-card h3 {
    font-size: 1.2em;
    margin-bottom: 5px;
    min-height: 2.4em; /* Ensure consistent height for titles (approx 2 lines) */
    display: -webkit-box;
    -webkit-line-clamp: 2; /* Limit title to 2 lines */
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}
.news-card p {
    font-size: 0.9em;
    color: #555;
    flex-grow: 1; /* Allows summary to take up available space */
    min-height: 4.5em; /* Ensure consistent height for summaries (approx 3 lines) */
    display: -webkit-box;
    -webkit-line-clamp: 3; /* Limit summary to 3 lines */
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-overflow: ellipsis;
}
.news-card a {
    text-decoration: none;
    color: #007bff;
    font-weight: bold;
    margin-top: auto; /* Pushes link to bottom */
}
.news-card a:hover {
    text-decoration: underline;
}

/* For horizontal scrolling category buttons (using st.radio) */
.stRadio > label {
    margin-right: 15px;
    padding: 8px 12px;
    border: 1px solid #007bff;
    border-radius: 20px;
    cursor: pointer;
    background-color: #f0f8ff;
    white-space: nowrap; /* Prevent buttons from wrapping */
}
.stRadio > label:hover {
    background-color: #e0f0ff;
}
/* Style for the selected radio button */
.stRadio [aria-checked="true"] > div:first-child { /* Targets the inner div that gets styled */
    background-color: #007bff !important; /* Force background color */
    color: white !important; /* Force text color */
    border-color: #007bff !important; /* Force border color */
}

/* Make the radio button container scrollable */
div[data-testid="stRadio"] > div {
    flex-wrap: nowrap; /* Prevent wrapping of radio buttons */
    overflow-x: auto; /* Enable horizontal scrolling */
    -webkit-overflow-scrolling: touch; /* For smoother scrolling on iOS */
    scrollbar-width: thin;
    scrollbar-color: #ccc #f1f1f1;
}
div[data-testid="stRadio"] > div::-webkit-scrollbar {
    height: 6px;
}
div[data-testid="stRadio"] > div::-webkit-scrollbar-thumb {
    background-color: #ccc;
    border-radius: 3px;
}
div[data-testid="stRadio"] > div::-webkit-scrollbar-track {
    background-color: #f1f1f1;
}

</style>
""", unsafe_allow_html=True)


st.title("భారత్ పల్స్ (BharatPulse) 🇮🇳")
st.markdown("మీ స్థానిక మరియు జాతీయ వార్తల వన్-స్టాప్ మూలం (Your one-stop source for local and national news)")

# Add "User Profile" to the main tabs
main_tabs = st.tabs(["📰 Headlines", "📍 My Location", "🔍 Search City", "🎙️ Voice Search", "👤 User Profile"])

# Initialize session state variables for user profile details
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'user_location' not in st.session_state:
    st.session_state.user_location = ""
if 'account_creation_date' not in st.session_state:
    st.session_state.account_creation_date = "N/A" # Could set to current date on first run

# --- Tab 1: Headlines ---
with main_tabs[0]:
    st.header("ప్రధాన వార్తలు (Headlines)")

    # Category selection for filtering
    st.markdown("##### వార్తల విభాగాలు (News Categories)")
    # Using st.radio with horizontal option for category selection
    selected_category_tl = st.radio(
        "విభాగం ఎంచుకోండి (Select Category):",
        options=list(TELUGU_CATEGORIES.keys()),
        index=0,
        horizontal=True, # Makes radio buttons horizontal and scrollable
        key="headlines_category_select"
    )
    # The actual category filtering logic would go here based on `selected_category_tl`
    # For now, it's a placeholder.
    st.markdown(f"**ఎంచుకున్న విభాగం:** {selected_category_tl}")

    st.markdown("---")

    all_articles = []
    with st.spinner("వార్తలను లోడ్ చేస్తోంది... (Loading news...)"):
        for source_name, rss_url in RSS_FEEDS.items():
            st.markdown(f"**{source_name} నుండి వార్తలు**")
            articles = get_news_from_rss(rss_url)
            
            for article in articles:
                # Simple heuristic to determine if translation is needed: check for ASCII alpha chars
                # This prevents trying to translate already Telugu titles or titles with mixed scripts
                if any(char.isalpha() and char.isascii() for char in article['title']):
                    article['translated_title'] = translate_en_to_te(article['title'])
                else:
                    article['translated_title'] = article['title'] # Assume it's already Telugu or mixed

            all_articles.extend(articles)
        
        if not all_articles:
            st.warning("వార్తలు లోడ్ చేయబడలేదు. దయచేసి మీ RSS ఫీడ్ URLలను తనిఖి చేయండి లేదా ఇంటర్నెట్ కనెక్షన్\u200cని తనిఖి చేయండి. (No news loaded. Please check your RSS feed URLs or internet connection.)")

    # Display news in a grid (swipe-style mock-up)
    if all_articles:
        num_cols = 3 # Number of columns for news cards
        rows = []
        for i in range(0, len(all_articles), num_cols):
            rows.append(all_articles[i:i + num_cols])

        for row_articles in rows:
            cols = st.columns(num_cols)
            for i, article in enumerate(row_articles):
                with cols[i]:
                    with st.container(): # Using container for card effect
                        st.markdown(f'<div class="news-card">', unsafe_allow_html=True)
                        if article['image_url']:
                            st.image(article['image_url'], use_column_width="always", caption="")
                        
                        # Display translated title first, with expander for original if translated
                        if article['translated_title'] and article['translated_title'] != article['title'] and article['translated_title'] != "Translation failed.":
                            st.markdown(f"### {article['translated_title']}")
                            with st.expander("Original Title"):
                                st.write(article['title'])
                        else:
                            st.markdown(f"### {article['title']}") # Display original if no translation or translation failed
                        
                        st.markdown(f"<p>{article['summary']}</p>", unsafe_allow_html=True) # Summary is already truncated in get_news_from_rss
                        st.markdown(f"[పూర్తిగా చదవండి (Read More)]({article['link']})", unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)


# --- Tab 2: My Location ---
with main_tabs[1]:
    st.header("📍 నా స్థానం (My Location)")

    user_city, user_state, user_country = get_user_location()
    st.write(f"మీ ప్రస్తుత స్థానం: **{user_city}, {user_state}, {user_country}**")

    st.subheader("నిజ-సమయ వాతావరణం + హెచ్చరికలు (Real-time Weather + Alerts)")
    if OPENWEATHERMAP_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY_HERE":
        st.error("దయచేసి OpenWeatherMap API కీని కాన్ఫిగర్ చేయండి. వాతావరణ సమాచారం అందుబాటులో లేదు. (Please configure your OpenWeatherMap API key. Weather data is not available.)")
    else:
        weather_data = get_current_weather(user_city, OPENWEATHERMAP_API_KEY)
        if weather_data:
            st.metric(label="ఉష్ణోగ్రత (Temperature)", value=f"{weather_data['temperature']:.1f}°C", delta=f"Feels like {weather_data['feels_like']:.1f}°C")
            st.write(f"**పరిస్థితి (Condition):** {weather_data['description']}")
            st.write(f"**తేమ (Humidity):** {weather_data['humidity']}%")
            st.write(f"**గాలి వేగం (Wind Speed):** {weather_data['wind_speed']:.1f} m/s")
            st.info(f"'{user_city}' కోసం ప్రస్తుత వాతావరణం. (Current weather for '{user_city}').")
        else:
            st.info(f"'{user_city}' కోసం వాతావరణ డేటా అందుబాటులో లేదు. (Weather data not available for '{user_city}').")

    st.subheader("స్థానిక వార్తలు (Local News)")
    # Placeholder for local news (e.g., fetch from specific local RSS or scrape, needs specific city-level data)
    st.info(f"'{user_city}' కు సంబంధించిన స్థానిక వార్తలు ఇక్కడ ప్రదర్శించబడతాయి. (Local news for '{user_city}' will be displayed here.)")
    
    # Placeholder for alerts (e.g., integrate with disaster management APIs or specific news sources)
    st.subheader("అలర్ట్స్ (Alerts)")
    st.info("స్థానిక హెచ్చరికలు ఇక్కడ ప్రదర్శించబడతాయి. (Local alerts will be displayed here.)")


# --- Tab 3: Search City ---
with main_tabs[2]:
    st.header("🔍 నగరాన్ని శోధించండి (Search City)")

    search_city = st.text_input("నగరం లేదా జిల్లా పేరును నమోదు చేయండి (Enter City or District Name):", "హైదరాబాద్", key="city_search_input")
    
    if st.button("వార్తలు శోధించండి (Search News)", key="search_city_button"):
        st.write(f"**'{search_city}'** కోసం వార్తలు లోడ్ అవుతున్నాయి... (Loading news for '{search_city}')...")
        
        # Display weather for the searched city
        st.subheader(f"వాతావరణం: {search_city} (Weather: {search_city})")
        if OPENWEATHERMAP_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY_HERE":
            st.error("దయచేసి OpenWeatherMap API కీని కాన్ఫిగర్ చేయండి. వాతావరణ సమాచారం అందుబాటులో లేదు. (Please configure your OpenWeatherMap API key. Weather data is not available.)")
        else:
            weather_data_search = get_current_weather(search_city, OPENWEATHERMAP_API_KEY)
            if weather_data_search:
                st.metric(label="ఉష్ణోగ్రత (Temperature)", value=f"{weather_data_search['temperature']:.1f}°C", delta=f"Feels like {weather_data_search['feels_like']:.1f}°C")
                st.write(f"**పరిస్థితి (Condition):** {weather_data_search['description']}")
                st.write(f"**తేమ (Humidity):** {weather_data_search['humidity']}%")
                st.write(f"**గాలి వేగం (Wind Speed):** {weather_data_search['wind_speed']:.1f} m/s")
            else:
                st.info(f"'{search_city}' కోసం వాతావరణ డేటా అందుబాటులో లేదు. (Weather data not available for '{search_city}').")

        st.subheader(f"స్థానిక వార్తలు: {search_city} (Local News: {search_city})")
        st.warning("ఈ ఫీచర్ కోసం స్థానిక వార్తా వనరులను అనుసంధానించడం అవసరం. (This feature requires integration with local news sources.)")
        st.info(f"'{search_city}' కు సంబంధించిన వార్తలు ఇక్కడ ప్రదర్శించబడతాయి. (News for '{search_city}' will be displayed here.)")


# --- Tab 4: Voice Search ---
with main_tabs[3]:
    voice_search_widget()

# --- Tab 5: User Profile (New Tab) ---
with main_tabs[4]:
    st.header("👤 యూజర్ ప్రొఫైల్ (User Profile)")
    st.markdown("మీ ప్రొఫైల్ వివరాలు మరియు యాప్ వినియోగ విశ్లేషణలు (Your profile details and app usage analytics)")

    # User Input for Personal Details
    st.subheader("వ్యక్తిగత వివరాలు (Personal Details)")
    
    # Use st.text_input with session state
    st.session_state.user_name = st.text_input(
        "పేరు (Name):", 
        value=st.session_state.user_name, 
        placeholder="మీ పేరు నమోదు చేయండి (Enter your name)",
        key="profile_name"
    )
    st.session_state.user_email = st.text_input(
        "ఇమెయిల్ (Email):", 
        value=st.session_state.user_email, 
        placeholder="మీ ఇమెయిల్ నమోదు చేయండి (Enter your email)",
        key="profile_email"
    )
    st.session_state.user_location = st.text_input(
        "స్థానం (Location):", 
        value=st.session_state.user_location, 
        placeholder="మీ స్థానం నమోదు చేయండి (Enter your location)",
        key="profile_location"
    )

    # You could also add a date picker for account creation or auto-set it on first entry
    if st.session_state.account_creation_date == "N/A" and st.session_state.user_name:
        import datetime
        st.session_state.account_creation_date = datetime.date.today().strftime("%B %d, %Y")

    st.write(f"**ఖాతా సృష్టించిన తేదీ (Account Created On):** {st.session_state.account_creation_date}")

    st.markdown("---")

    # Mock User Activity/Preferences
    st.subheader("యాప్ వినియోగం (App Usage)")
    
    st.markdown("###### మీ వీక్షణలు (Your Views):")
    st.info("ఈ విభాగానికి వాస్తవ వినియోగదారు డేటా నిల్వ అవసరం. ఇవి ఉదాహరణ పాయింట్లు. (This section requires actual user data storage. These are example points.)")
    
    st.markdown("""
    <ul>
        <li><b>చూసిన వార్తల సంఖ్య (Articles Viewed):</b> 150+</li>
        <li><b>అత్యంత ఎక్కువగా చూసిన వర్గం (Most Viewed Category):</b> రాజకీయాలు (Politics)</li>
        <li><b>అత్యంత ఎక్కువగా చూసిన మూలం (Most Viewed Source):</b> సాక్షి (Sakshi)</li>
        <li><b>చివరిగా చూసిన వార్త (Last Viewed Article):</b> "హైదరాబాద్‌లో కొత్త వంతెన నిర్మాణం" (New Bridge Construction in Hyderabad)</li>
        <li><b>అనువాదాలను ఉపయోగించిన సంఖ్య (Translations Used):</b> 50 సార్లు (50 times)</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("ప్రాధాన్యతలు (Preferences)")
    st.write("వార్తల వర్గం ప్రాధాన్యతలు (Preferred News Categories):")
    preferred_categories = st.multiselect(
        "మీ ప్రాధాన్యత గల వర్గాలను ఎంచుకోండి (Select your preferred categories):",
        options=list(TELUGU_CATEGORIES.keys()),
        default=["జాతీయ వార్తలు", "రాష్ట్ర వార్తలు", "వ్యాపారం"]
    )
    st.write(f"మీరు ఎంచుకున్నవి: {', '.join(preferred_categories)}")

    st.write("వార్తా మూలాల ప్రాధాన్యతలు (Preferred News Sources):")
    preferred_sources = st.multiselect(
        "మీ ప్రాధాన్యత గల వార్తా మూలాలను ఎంచుకోండి (Select your preferred news sources):",
        options=list(RSS_FEEDS.keys()),
        default=["Sakshi"]
    )
    st.write(f"మీరు ఎంచుకున్నవి: {', '.join(preferred_sources)}")

    st.info("ఈ ప్రాధాన్యతలు భవిష్యత్తులో మీ వార్తల ఫీడ్‌ను వ్యక్తిగతీకరించడానికి ఉపయోగించబడతాయి. (These preferences will be used to personalize your news feed in the future.)")

st.sidebar.title("భారత్ పల్స్ - సెట్టింగ్‌లు")
st.sidebar.info("ఇక్కడ మీరు యాప్ సెట్టింగ్‌లను కాన్ఫిగర్ చేయవచ్చు. (Here you can configure app settings.)")

if st.sidebar.button("రీఫ్రెష్ వార్తలు", key="refresh_news_button"):
    st.cache_data.clear() # Clear all data caches (news, location, weather)
    st.cache_resource.clear() # Clear model cache (translation model)
    st.rerun() # Rerun the app to load fresh data
    st.sidebar.success("వార్తలు రీఫ్రెష్ చేయబడ్డాయి!")

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 భారత్ పల్స్")
