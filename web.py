import os
import joblib
import numpy as np
import streamlit as st
import google.generativeai as genai
from streamlit_extras.stylable_container import stylable_container
import requests
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #BB86FC;
        font-weight: 600;
    }
    
    /* Form elements */
    .stSelectbox, .stNumberInput, .stSlider {
        background-color: #1E1E1E;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #BB86FC;
        color: #121212;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #9D74D0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Card/Box styling */
    .bento-box {
        background-color: #1E1E1E;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 8px 8px 0 0;
        color: #f0f0f0;
        padding: 10px 24px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #BB86FC;
        color: #121212;
        font-weight: 600;
    }
    
    /* Success message styling */
    .success-box {
        background-color: #1E1E1E;
        border-left: 5px solid #03DAC6;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    /* Warning message styling */
    .stWarning {
        background-color: rgba(255, 171, 0, 0.1);
        border-left: 5px solid #FFAB00;
    }
    
    /* Error message styling */
    .stError {
        background-color: rgba(207, 102, 121, 0.1);
        border-left: 5px solid #CF6679;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_resources():
    resources = {}
    file_path = './Models/laptop_dataset.pkl'
    model_path = './Models/XGBoost_regressor_model.pkl'
    encoder_path = './Models/encoder.pkl'
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                resources['df'] = joblib.load(file)
        else:
            st.error(f"File not found: {file_path}")
            resources['df'] = None
            
        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                resources['model'] = joblib.load(file)
        else:
            st.error(f"File not found: {model_path}")
            resources['model'] = None
            
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as file:
                resources['encoder'] = joblib.load(file)
        else:
            st.error(f"File not found: {encoder_path}")
            resources['encoder'] = None
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        
    return resources

resources = load_resources()
df = resources.get('df')
model = resources.get('model')
encoder = resources.get('encoder')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure Gemini AI (replace with your API key)
def configure_gemini():
    try:
        # Replace with your actual API key
        genai.configure(api_key=os.getenv('GEMINI_API_KEY', 'your-api-key-here'))
        return genai.GenerativeModel('gemini-1.0-pro-latest')
    except Exception as e:
        st.error(f"Error configuring Gemini AI: {e}")
        return None

# Web scraping function for laptop recommendations
def scrape_laptop_data(budget, requirements):
    try:
        # This is a placeholder. In a real application, you would implement proper web scraping
        # of laptop data from review sites or e-commerce platforms
        st.info("Note: In a production environment, this would scrape real-time data from trusted sources.")
        
        # For demonstration, we'll return sample data based on the budget
        if budget < 50000:
            return [
                {"name": "Acer Aspire 5", "price": 45999, "specs": "i5, 8GB RAM, 512GB SSD"},
                {"name": "Lenovo IdeaPad 3", "price": 42999, "specs": "Ryzen 5, 8GB RAM, 512GB SSD"}
            ]
        elif budget < 80000:
            return [
                {"name": "HP Pavilion 15", "price": 75999, "specs": "i7, 16GB RAM, 512GB SSD"},
                {"name": "Dell Inspiron 15", "price": 72999, "specs": "Ryzen 7, 16GB RAM, 512GB SSD"}
            ]
        else:
            return [
                {"name": "ASUS ROG Strix G15", "price": 95999, "specs": "i7, 16GB RAM, 1TB SSD, RTX 3060"},
                {"name": "MacBook Air M2", "price": 102999, "specs": "M2, 8GB RAM, 256GB SSD"}
            ]
    except Exception as e:
        st.error(f"Error scraping data: {e}")
        return []

# Main app layout with tabs
tab1, tab2 = st.tabs(["💰 Price Prediction", "💬 Expert Advice"])

# Tab 1: Price Prediction
with tab1:
    st.markdown("<h1 style='text-align: center; color: #BB86FC;'>Laptop Price Predictor</h1>", unsafe_allow_html=True)
    
    if df is not None:
        with st.form(key='specs_form'):
            col1, col2 = st.columns(2)
            
            with col1:
                with stylable_container(
                    key="brand_box",
                    css_styles="""
                        {
                            background-color: #1E1E1E;
                            border-radius: 10px;
                            padding: 20px;
                            margin-bottom: 20px;
                        }
                    """
                ):
                    st.subheader("🏢 Brand & Type")
                    company = st.selectbox("Select a brand", ["Select"] + list(df['Company'].unique()))
                    type_name = st.selectbox("Select Laptop type", ["Select"] + list(df['TypeName'].unique()))
                    weight = st.number_input('Weight (kg)', step=0.1, value=1.4, min_value=0.5, max_value=5.0)
                
                with stylable_container(
                    key="display_box",
                    css_styles="""
                        {
                            background-color: #1E1E1E;
                            border-radius: 10px;
                            padding: 20px;
                            margin-bottom: 20px;
                        }
                    """
                ):
                    st.subheader("🖥️ Display Features")
                    display_col1, display_col2 = st.columns(2)
                    with display_col1:
                        touchscreen = st.checkbox("Touchscreen")
                    with display_col2:
                        ips = st.checkbox("IPS Panel")
                    
                    screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1, value=15.6)
                    resolution = st.selectbox("Resolution", [
                        "Select", '1920x1080', '1366x768', '1600x900', 
                        '3480x2160', '3200x1800', '2800x1800', 
                        '2560x1440', '2304x1440'
                    ])
            
            with col2:
                with stylable_container(
                    key="performance_box",
                    css_styles="""
                        {
                            background-color: #1E1E1E;
                            border-radius: 10px;
                            padding: 20px;
                            margin-bottom: 20px;
                        }
                    """
                ):
                    st.subheader("⚡ Performance")
                    ram = st.slider("RAM Size (GB)", min_value=2, max_value=64, step=2, value=8)
                    processor = st.selectbox("Processor", ["Select"] + list(df['Processor'].unique()))
                    gpu = st.selectbox("GPU", ["No GPU"] + list(df['Gpu'].unique()))
                
                with stylable_container(
                    key="storage_box",
                    css_styles="""
                        {
                            background-color: #1E1E1E;
                            border-radius: 10px;
                            padding: 20px;
                            margin-bottom: 20px;
                        }
                    """
                ):
                    st.subheader("💾 Storage")
                    storage_col1, storage_col2 = st.columns(2)
                    with storage_col1:
                        hdd = st.selectbox("HDD Size (GB)", [0, 128, 256, 512, 1024, 2048])
                    with storage_col2:
                        ssd = st.selectbox("SSD Size (GB)", [0, 8, 128, 256, 512, 1024])
                
                with stylable_container(
                    key="os_box",
                    css_styles="""
                        {
                            background-color: #1E1E1E;
                            border-radius: 10px;
                            padding: 20px;
                        }
                    """
                ):
                    st.subheader("🖥️ Operating System")
                    os_laptop = st.selectbox("Operating System", ["Select"] + list(df['OpSys'].unique()))
            
            predict_btn = st.form_submit_button("Predict Price")
            
            if predict_btn:
                if (company != "Select" and type_name != "Select" and resolution != "Select" 
                    and processor != "Select" and os_laptop != "Select"):
                    
                    # Calculate PPI
                    x = resolution.split('x')[0]
                    y = resolution.split('x')[1]
                    ppi = (int(x)**2 + int(y)**2)**0.5 / screen_size
                    
                    # Prepare query
                    query = np.array([
                        str(company), str(type_name), int(ram), str(gpu), str(os_laptop), 
                        float(weight), int(touchscreen), int(ips), float(ppi), 
                        str(processor), int(hdd), int(ssd)
                    ])
                    
                    # Make prediction
                    try:
                        query = encoder.transform([query])
                        result = model.predict(query)
                        result = np.exp(result)
                        
                        # Display result in a stylish box
                        st.markdown(f"""
                        <div class='success-box'>
                            <h2 style='color: #03DAC6; margin-bottom: 10px;'>Price Prediction</h2>
                            <p style='font-size: 24px; font-weight: bold;'>
                                The estimated price of the laptop is ₹{result[0]:,.2f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show specification summary
                        st.markdown("<h3>Specification Summary</h3>", unsafe_allow_html=True)
                        spec_col1, spec_col2 = st.columns(2)
                        
                        with spec_col1:
                            with stylable_container(
                                key="spec_summary_1",
                                css_styles="""
                                    {
                                        background-color: #1E1E1E;
                                        border-radius: 10px;
                                        padding: 20px;
                                    }
                                """
                            ):
                                st.markdown(f"""
                                - **Brand**: {company}
                                - **Type**: {type_name}
                                - **Processor**: {processor}
                                - **RAM**: {ram} GB
                                """)
                        
                        with spec_col2:
                            with stylable_container(
                                key="spec_summary_2",
                                css_styles="""
                                    {
                                        background-color: #1E1E1E;
                                        border-radius: 10px;
                                        padding: 20px;
                                    }
                                """
                            ):
                                st.markdown(f"""
                                - **Display**: {screen_size}" {resolution} {'(Touchscreen)' if touchscreen else ''}
                                - **Storage**: {hdd} GB HDD + {ssd} GB SSD
                                - **GPU**: {gpu}
                                - **OS**: {os_laptop}
                                """)
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                else:
                    st.warning("Please fill in all required fields")
    else:
        st.error("Data not available. Please check the file path and try again.")

# Tab 2: Expert Advice
with tab2:
    st.markdown("<h1 style='text-align: center; color: #BB86FC;'>Laptop Expert Advisor</h1>", unsafe_allow_html=True)
    
    # Initialize Gemini AI
    gemini_model = configure_gemini()
    
    if gemini_model:
        # Chat interface
        st.markdown("""
        <div style='background-color: #1E1E1E; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
            <p>Welcome to the Laptop Expert Advisor! Ask me anything about laptops, and I'll provide expert advice.</p>
            <p>Example questions:</p>
            <ul>
                <li>What's the difference between SSD and HDD?</li>
                <li>Recommend a laptop for video editing under ₹80,000</li>
                <li>What specs do I need for gaming?</li>
                <li>Which processor is better for programming?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Get user input
        if prompt := st.chat_input("Ask your laptop questions here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Check if it's a recommendation request
            is_recommendation = any(keyword in prompt.lower() for keyword in ["recommend", "suggest", "buy", "purchase", "budget"])
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                try:
                    if is_recommendation:
                        # Extract budget from prompt if present
                        import re
                        budget_match = re.search(r'(\d+),?(\d+)?', prompt)
                        budget = 70000  # Default budget
                        if budget_match:
                            budget_str = budget_match.group(0).replace(',', '')
                            budget = int(budget_str)
                        
                        # Get recommendations from Gemini
                        gemini_response = gemini_model.generate_content(
                            f"Provide laptop recommendations for this query: '{prompt}'. Focus on technical specifications, " 
                            f"price points, and explain why these laptops are good for the user's needs. Be specific and concise."
                        ).text
                        
                        # Get additional recommendations from web scraping
                        laptop_recommendations = scrape_laptop_data(budget, prompt)
                        
                        # Combine the responses
                        response = f"{gemini_response}\n\n### Current Market Recommendations:\n"
                        for laptop in laptop_recommendations:
                            response += f"- **{laptop['name']}** - ₹{laptop['price']:,} - {laptop['specs']}\n"
                    else:
                        # Get a regular response from Gemini
                        response = gemini_model.generate_content(
                            f"Answer this laptop-related question as an expert: '{prompt}'. " 
                            f"Provide accurate, helpful information with technical details where appropriate."
                        ).text
                    
                    message_placeholder.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Show budget-based recommendation form
        with stylable_container(
            key="recommendation_form",
            css_styles="""
                {
                    background-color: #1E1E1E;
                    border-radius: 10px;
                    padding: 20px;
                    margin-top: 30px;
                }
            """
        ):
            st.subheader("Quick Recommendation Tool")
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                budget = st.number_input("Your Budget (₹)", min_value=20000, max_value=200000, value=70000, step=5000)
                use_case = st.selectbox("Primary Use", [
                    "General/Everyday Use", 
                    "Gaming", 
                    "Video Editing/Graphic Design",
                    "Programming/Development",
                    "Business/Office Work",
                    "Student"
                ])
            
            with rec_col2:
                portability = st.slider("Portability Importance", 1, 10, 5)
                brand_preference = st.multiselect("Brand Preference (Optional)", 
                    ["No Preference", "HP", "Dell", "Lenovo", "ASUS", "Acer", "Apple", "MSI"])
            
            if st.button("Get Personalized Recommendations"):
                # Create a prompt based on the form inputs
                recommendation_prompt = (
                    f"Recommend laptops for a budget of ₹{budget} for {use_case}. "
                    f"Portability importance: {portability}/10. "
                )
                
                if brand_preference and "No Preference" not in brand_preference:
                    recommendation_prompt += f"Preferred brands: {', '.join(brand_preference)}. "
                
                # Add this to the chat
                st.session_state.messages.append({"role": "user", "content": recommendation_prompt})
                
                with st.chat_message("user"):
                    st.markdown(recommendation_prompt)
                
                with st.chat_message("assistant"):
                    rec_placeholder = st.empty()
                    
                    try:
                        # Get recommendations from Gemini
                        gemini_response = gemini_model.generate_content(
                            f"You are a laptop expert. {recommendation_prompt} "
                            f"Provide 3 specific laptop models with exact specifications and price ranges. "
                            f"Format your response with clear headings and bullet points for each recommendation. "
                            f"For each recommendation, explain why it's a good fit for the use case."
                        ).text
                        
                        # Get additional recommendations from web scraping
                        laptop_recommendations = scrape_laptop_data(budget, use_case)
                        
                        # Combine the responses
                        response = f"{gemini_response}\n\n### Market Availability:\n"
                        for laptop in laptop_recommendations:
                            response += f"- **{laptop['name']}** - ₹{laptop['price']:,} - {laptop['specs']}\n"
                        
                        rec_placeholder.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        rec_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.error("Could not initialize Gemini AI. Please check your API key and try again.")
