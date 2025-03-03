import os
import joblib
import numpy as np
import streamlit as st
import time
import google.generativeai as genai
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.stylable_container import stylable_container
import base64

# Configure page
st.set_page_config(
    page_title="Laptech AI",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for tech theme
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #111111;
        color: #00FF7F;
    }
    /* Tech font */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    /* Custom header */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #222222;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #333333;
        color: #00FF7F;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #555555 !important;
        color: #00FFFF !important;
    }
    /* Form elements */
    .stButton button {
        background-color: #00FF7F;
        color: black;
        border: none;
        border-radius: 4px;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox, .stSlider {
        background-color: #222222;
        color: #00FF7F;
        border-color: #444444;
    }
    /* Sidebar */
    .css-1d391kg, .css-1e5imcs {
        background-color: #222222;
    }
    /* Success messages */
    .element-container div[data-testid="stAlert"] {
        background-color: #004422;
        border: 1px solid #00FF7F;
        color: #00FF7F;
    }
    /* Typewriter effect container */
    .typewriter-container {
        border: 1px solid #00FF7F;
        border-radius: 4px;
        padding: 10px;
        background-color: #222222;
        min-height: 100px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_resources():
    resources = {}
    file_path = './Models/laptop_dataset.pkl'
    model_path = './Models/XGBoost_regressor_model.pkl'
    encoder_path = './Models/encoder.pkl'
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            resources['df'] = joblib.load(file)
    else:
        st.error(f"File not found: {file_path}")
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            resources['model'] = joblib.load(file)
    else:
        st.error(f"File not found: {model_path}")
      
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as file:
            resources['encoder'] = joblib.load(file)
    else:
        st.error(f"File not found: {encoder_path}")
    
    return resources

resources = load_resources()

# Sidebar
with st.sidebar:
    # Logo (placeholder - replace with actual logo path)
    st.image("https://static.vecteezy.com/system/resources/thumbnails/013/441/060/small_2x/open-laptop-icon-illustration-png.png", width=150)
    
    st.markdown("<h1 style='color:#00FF7F;'>Laptech AI</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style='color:#00FF7F;'>Laptech AI is a cutting-edge tool that leverages machine learning to predict laptop prices based on specifications. Our AI assistant also provides expert advice to help you make informed purchasing decisions in the ever-evolving tech market.</p>
    """, unsafe_allow_html=True)
    
    add_vertical_space(2)
    st.markdown("<p style='color:#00FF7F; text-align:center;'>© 2025 Laptech AI</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#00FF7F; text-align:center;'>Made with 💚 by Mainak</p>", unsafe_allow_html=True)

# Main content with tabs
tab1, tab2 = st.tabs(["💲 Price Prediction", "🤖 Expert Advice"])

# Tab 1: Price Prediction
with tab1:
    colored_header(
        label="Laptop Price Predictor",
        description="Get accurate price estimates based on specifications",
        color_name="green-70"
    )
    
    if 'df' in resources:
        df = resources['df']
        model = resources['model']
        encoder = resources['encoder']
        
        with stylable_container(
            key="specs_container",
            css_styles="""
                {
                    background-color: #222222;
                    border-radius: 10px;
                    padding: 20px;
                    border: 1px solid #444444;
                }
            """
        ):
            with st.form(key='specs_form'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h3 style='color:#00FFFF;'>Basic Information</h3>", unsafe_allow_html=True)
                    company = st.selectbox("Brand", ["Select"] + list(df['Company'].unique()))
                    type = st.selectbox("Laptop Type", ["Select"] + list(df['TypeName'].unique()))
                    weight = st.number_input('Weight (kg)', step=0.1, value=1.4, min_value=0.5, max_value=5.0)
                    
                    st.markdown("<h3 style='color:#00FFFF;'>RAM & Storage</h3>", unsafe_allow_html=True)
                    ram = st.slider("RAM (GB)", min_value=2, max_value=64, step=2, value=8)
                    hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024, 2048], index=0)
                    ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1024], index=0)
                
                with col2:
                    st.markdown("<h3 style='color:#00FFFF;'>Display</h3>", unsafe_allow_html=True)
                    screen_size = st.number_input('Screen Size (inches)', min_value=12.0, max_value=20.0, step=0.1, value=15.6)
                    resolution = st.selectbox("Resolution", ['1920x1080', '1366x768', '1600x900', '3480x2160', '3200x1800', '2800x1800', '2560x1440', '2304x1440'])
                    touchscreen = st.checkbox("Touchscreen")
                    ips = st.checkbox("IPS Panel")
                    
                    st.markdown("<h3 style='color:#00FFFF;'>Performance & OS</h3>", unsafe_allow_html=True)
                    processor = st.selectbox("Processor", list(df['Processor'].unique()))
                    gpu = st.selectbox("GPU", ["No GPU"] + list(df['Gpu'].unique()))
                    os_laptop = st.selectbox("Operating System", list(df['OpSys'].unique()))
                
                predict_btn = st.form_submit_button(label='📊 PREDICT PRICE', use_container_width=True)
                
                if predict_btn:
                    if company != "Select" and type != "Select":
                        with st.spinner("Calculating price..."):
                            # Calculate PPI
                            x = resolution.split('x')[0]
                            y = resolution.split('x')[1]
                            ppi = (int(x)**2 + int(y)**2)**0.5 / screen_size
                            
                            # Prepare query
                            query = np.array([
                                str(company), str(type), int(ram), str(gpu), 
                                str(os_laptop), float(weight), int(touchscreen), 
                                int(ips), float(ppi), processor, int(hdd), int(ssd)
                            ])
                            
                            # Transform and predict
                            query = encoder.transform([query])
                            result = model.predict(query)
                            result = np.exp(result)
                            
                            # Display result
                            st.balloons()
                            st.success(f"### Estimated Price: ₹{result[0]:,.2f}")
                    else:
                        st.warning("⚠️ Please fill in all required fields")
    else:
        st.error("❌ Data not available. Please check the file paths and try again.")

# Tab 2: Expert Advice
with tab2:
    colored_header(
        label="AI Laptop Expert",
        description="Get personalized advice for your laptop needs",
        color_name="green-70"
    )
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Set up Gemini API - You'll need to add your API key in the code or environment variables
    try:
        # Get API key from environment variable
        api_key = st.secrets["GEMINI_API_KEY"]
        if api_key:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            is_configured = True
        else:
            # Fallback message - this will only be visible to developers, not end users
            st.info("Note to developer: Set the GEMINI_API_KEY environment variable")
            is_configured = False
            model = None
    except Exception as e:
        st.error("Error initializing AI assistant")
        is_configured = False
        model = None
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about laptops (e.g., 'Which laptop is best for gaming?')"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate AI response
        if is_configured:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                # Prepare prompt with context
                context = """
                You are a laptop expert advisor. Provide helpful, accurate advice about laptops.
                Focus on the user's specific needs and interests. Your advice should be technical
                but easy to understand. Recommend specific features based on use cases like gaming,
                productivity, design work, etc. Keep responses concise and helpful.
                """
                
                try:
                    response = model.generate_content(context + "\n\nUser question: " + prompt)
                    
                    # Simulate typewriter effect
                    if hasattr(response, 'text'):
                        response_text = response.text
                        for chunk in response_text.split():
                            full_response += chunk + " "
                            time.sleep(0.05)  # Adjust speed as needed
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        message_placeholder.markdown("Sorry, I couldn't generate a response.")
                except Exception as e:
                    message_placeholder.markdown(f"Error generating response: {str(e)}")
        else:
            with st.chat_message("assistant"):
                st.markdown("AI assistant is currently unavailable. Please try again later.")
