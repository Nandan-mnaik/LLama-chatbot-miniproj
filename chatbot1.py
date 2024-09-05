import streamlit as st
import replicate
import os

st.set_page_config(page_title="Llama 2-7B Chatbot")
#css stylingf link
with open('styles.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Sidebar for API token input and parameters
with st.sidebar:
    st.title(' Llama Chatbot')

    # Enter Replicate API token
    replicate_api = st.text_input('Enter Replicate API token:', type='password')

    # Model parameters
    temperature = st.slider('Temperature', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
    top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.slider('Max Length', min_value=32, max_value=512, value=128, step=32)

# Store the Replicate API token in the environment variable
if replicate_api:
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Define the Replicate Llama 2-7B model version (this is the specific ID for Llama 2-7B hosted on Replicate)
llama_2_7b_model = "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea"

# Set up chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": " How may I assist you today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear chat history function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": " How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to call Replicate Llama 2-7B model API
def generate_response(prompt_input):
    # Ensure Replicate API token is set
    if not replicate_api:
        st.warning("Please enter a valid Replicate API token.")
        return None
    
    # Make the API request to Replicate's Llama 2-7B model
    try:
        output = replicate.run(
            llama_2_7b_model,
            input={
                "prompt": prompt_input,
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "repetition_penalty": 1.0
            }
        )
        return output
    except Exception as e:
        st.error(f"Error calling Replicate API: {e}")
        return None

# User prompt input
prompt = st.chat_input("Enter your message here...")

# Handle user input and generate response
if prompt and replicate_api:
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            if response:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
#End of program