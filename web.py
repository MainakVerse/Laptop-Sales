import os
import joblib
import numpy as np
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS with improved styling and fixes for width issues
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
    
    /* Fix width issues for form elements */
    div[data-baseweb="select"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    div[data-baseweb="base-input"] {
        width: 100% !important;
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
    
    /* Card styling */
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
    
    /* Message styling */
    .success-box {
        background-color: #1E1E1E;
        border-left: 5px solid #03DAC6;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .warning-box {
        background-color: #1E1E1E;
        border-left: 5px solid #FFAB00;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .error-box {
        background-color: #1E1E1E;
        border-left: 5px solid #CF6679;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data with improved error handling
@st.cache_resource
def load_resources():
    resources = {}
    file_paths = {
        'df': './Models/laptop_dataset.pkl',
        'model': './Models/XGBoost_regressor_model.pkl',
        'encoder': './Models/encoder.pkl'
    }
    
    for key, path in file_paths.items():
        try:
            if os.path.exists(path):
                resources[key] = joblib.load(path)
            else:
                st.error(f"File not found: {path}")
                resources[key] = None
        except Exception as e:
            st.error(f"Error loading {key}: {e}")
            resources[key] = None
            
    return resources

# Configure Gemini AI
@st.cache_resource
def configure_gemini():
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            st.warning("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
            return None
            
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-1.0-pro')
    except Exception as e:
        st.error(f"Error configuring Gemini AI: {e}")
        return None

# Get laptop recommendations based on budget and requirements
def get_laptop_recommendations(budget, requirements):
    try:
        # In a real app, this would connect to a real-time data source
        st.info("Note: In a production environment, this would fetch real-time data from trusted sources.")
        
        # Example data structured by budget range
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
        st.error(f"Error getting recommendations: {e}")
        return []

# Initialize resources
resources = load_resources()
df = resources.get('df')
model = resources.get('model')
encoder = resources.get('encoder')

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main app layout with tabs
tab1, tab2 = st.tabs(["💰 Price Prediction", "💬 Expert Advice"])

# Tab 1: Price Prediction
with tab1:
    st.markdown("<h1 style='text-align: center; color: #BB86FC;'>Laptop Price Predictor</h1>", unsafe_allow_html=True)
    
    if df is not None and model is not None and encoder is not None:
        # Create a centered container with appropriate spacing
        col_spacer1, form_col, col_spacer2 = st.columns([1, 10, 1])
        
        with form_col:
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
                        company = st.selectbox("Select a brand", ["Select"] + sorted(list(df['Company'].unique())), key="brand_select")
                        type_name = st.selectbox("Select Laptop type", ["Select"] + sorted(list(df['TypeName'].unique())), key="type_select")
                        weight = st.number_input('Weight (kg)', step=0.1, value=1.4, min_value=0.5, max_value=5.0, key="weight_input")
                    
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
                            touchscreen = st.checkbox("Touchscreen", key="touch_check")
                        with display_col2:
                            ips = st.checkbox("IPS Panel", key="ips_check")
                        
                        screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, step=0.1, value=15.6, key="screen_input")
                        
                        # Sort resolutions for better UX
                        resolutions = sorted([
                            '1920x1080', '1366x768', '1600x900', 
                            '3480x2160', '3200x1800', '2800x1800', 
                            '2560x1440', '2304x1440'
                        ])
                        resolution = st.selectbox("Resolution", ["Select"] + resolutions, key="res_select")
                
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
                        ram = st.slider("RAM Size (GB)", min_value=2, max_value=64, step=2, value=8, key="ram_slider")
                        processor = st.selectbox("Processor", ["Select"] + sorted(list(df['Processor'].unique())), key="proc_select")
                        gpu = st.selectbox("GPU", ["No GPU"] + sorted(list(df['Gpu'].unique())), key="gpu_select")
                    
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
                            hdd = st.selectbox("HDD Size (GB)", [0, 128, 256, 512, 1024, 2048], key="hdd_select")
                        with storage_col2:
                            ssd = st.selectbox("SSD Size (GB)", [0, 8, 128, 256, 512, 1024], key="ssd_select")
                    
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
                        os_laptop = st.selectbox("Operating System", ["Select"] + sorted(list(df['OpSys'].unique())), key="os_select")
                
                # Center the button
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    predict_btn = st.form_submit_button("Predict Price")
                
                if predict_btn:
                    if (company != "Select" and type_name != "Select" and resolution != "Select" 
                        and processor != "Select" and os_laptop != "Select"):
                        
                        try:
                            # Calculate PPI
                            x_res, y_res = map(int, resolution.split('x'))
                            ppi = (x_res**2 + y_res**2)**0.5 / screen_size
                            
                            # Prepare query with proper types
                            query = np.array([
                                company, type_name, int(ram), 
                                gpu if gpu != "No GPU" else "", os_laptop, 
                                float(weight), int(touchscreen), int(ips), float(ppi), 
                                processor, int(hdd), int(ssd)
                            ])
                            
                            # Make prediction
                            query = encoder.transform([query])
                            result = model.predict(query)
                            predicted_price = np.exp(result)[0]
                            
                            # Display result in a stylish box
                            st.markdown(f"""
                            <div class='success-box'>
                                <h2 style='color: #03DAC6; margin-bottom: 10px;'>Price Prediction</h2>
                                <p style='font-size: 24px; font-weight: bold;'>
                                    The estimated price of the laptop is ₹{predicted_price:,.2f}
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
        st.error("Required models or data not available. Please check the file paths and try again.")

# Tab 2: Expert Advice
with tab2:
    st.markdown("<h1 style='text-align: center; color: #BB86FC;'>Laptop Expert Advisor</h1>", unsafe_allow_html=True)
    
    # Create a centered container
    chat_col1, chat_col2, chat_col3 = st.columns([1, 10, 1])
    
    with chat_col2:
        # Initialize Gemini AI
        gemini_model = configure_gemini()
        
        if gemini_model:
            # Chat interface with welcome message
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
                
                # Determine if it's a recommendation request
                is_recommendation = any(keyword in prompt.lower() for keyword in ["recommend", "suggest", "buy", "purchase", "budget"])
                
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    try:
                        # Handle recommendation requests
                        if is_recommendation:
                            # Extract budget from prompt
                            import re
                            budget_match = re.search(r'(\d+),?(\d+)?', prompt)
                            budget = 70000  # Default budget
                            if budget_match:
                                budget_str = budget_match.group(0).replace(',', '')
                                budget = int(budget_str)
                            
                            # Get recommendations from Gemini
                            safety_settings = [
                                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                            ]
                            
                            gemini_response = gemini_model.generate_content(
                                f"Provide laptop recommendations for this query: '{prompt}'. Focus on technical specifications, " 
                                f"price points, and explain why these laptops are good for the user's needs. Be specific and concise.",
                                safety_settings=safety_settings
                            ).text
                            
                            # Get market recommendations
                            laptop_recommendations = get_laptop_recommendations(budget, prompt)
                            
                            # Combine responses
                            response = f"{gemini_response}\n\n### Current Market Recommendations:\n"
                            for laptop in laptop_recommendations:
                                response += f"- **{laptop['name']}** - ₹{laptop['price']:,} - {laptop['specs']}\n"
                        else:
                            # Get a regular response for non-recommendation queries
                            response = gemini_model.generate_content(
                                f"Answer this laptop-related question as an expert: '{prompt}'. " 
                                f"Provide accurate, helpful information with technical details where appropriate."
                            ).text
                        
                        # Display the response
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
                
                # Create the recommendation form
                with st.form(key='recommendation_form_inner'):
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        budget = st.number_input("Your Budget (₹)", min_value=20000, max_value=200000, value=70000, step=5000, key="budget_input")
                        use_case = st.selectbox("Primary Use", [
                            "General/Everyday Use", 
                            "Gaming", 
                            "Video Editing/Graphic Design",
                            "Programming/Development",
                            "Business/Office Work",
                            "Student"
                        ], key="use_case_select")
                    
                    with rec_col2:
                        portability = st.slider("Portability Importance", 1, 10, 5, key="portability_slider")
                        brand_preference = st.multiselect("Brand Preference (Optional)", 
                            ["No Preference", "HP", "Dell", "Lenovo", "ASUS", "Acer", "Apple", "MSI"], 
                            default=["No Preference"],
                            key="brand_multi")
                    
                    # Center the button
                    _, rec_btn_col, _ = st.columns([1, 2, 1])
                    with rec_btn_col:
                        submit_rec = st.form_submit_button("Get Personalized Recommendations")
                    
                    # Handle recommendation form submission
                    if submit_rec:
                        # Create a prompt based on the form inputs
                        recommendation_prompt = (
                            f"Recommend laptops for a budget of ₹{budget} for {use_case}. "
                            f"Portability importance: {portability}/10. "
                        )
                        
                        if brand_preference and "No Preference" not in brand_preference:
                            recommendation_prompt += f"Preferred brands: {', '.join(brand_preference)}. "
                        
                        # Add to chat history
                        st.session_state.messages.append({"role": "user", "content": recommendation_prompt})
                        
                        # Using st.spinner to show a loading indicator
                        with st.spinner("Getting recommendations..."):
                            try:
                                # Get recommendations from Gemini
                                gemini_response = gemini_model.generate_content(
                                    f"You are a laptop expert. {recommendation_prompt} "
                                    f"Provide 3 specific laptop models with exact specifications and price ranges. "
                                    f"Format your response with clear headings and bullet points for each recommendation. "
                                    f"For each recommendation, explain why it's a good fit for the use case."
                                ).text
                                
                                # Get market recommendations
                                laptop_recommendations = get_laptop_recommendations(budget, use_case)
                                
                                # Combine responses
                                response = f"{gemini_response}\n\n### Market Availability:\n"
                                for laptop in laptop_recommendations:
                                    response += f"- **{laptop['name']}** - ₹{laptop['price']:,} - {laptop['specs']}\n"
                                
                                # Add to chat history
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                
                                # Force a rerun to update the chat history
                                st.experimental_rerun()
                            except Exception as e:
                                error_msg = f"Sorry, I encountered an error: {str(e)}"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                # Force a rerun to update the chat history
                                st.experimental_rerun()
        else:
            st.warning("Gemini AI is not configured. Please check your API key and try again.")
