
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Streamlit UI
st.title("🌍 AI-Powered Travel Planner")
st.subheader("Find the best travel options for your trip")

# User Inputs
source = st.text_input("Enter Source Location:")
destination = st.text_input("Enter Destination:")

# Load API Key
try:
    with open("key.txt") as f:
        key = f.read().strip()
except FileNotFoundError:
    st.error("API key file not found. Please add 'key.txt' with your API key.")
    st.stop()

# Initialize Google GenAI Model
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=key)

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["source", "destination"],
    template="Find optimal travel options from {source} to {destination}, including cab, train, bus, and flights with estimated cost."
)

# Define Output Schema
response_schemas = [
    ResponseSchema(name="travel_mode", description="Mode of transport (cab, train, bus, flight)"),
    ResponseSchema(name="estimated_cost", description="Estimated cost of travel in USD"),
    ResponseSchema(name="duration", description="Approximate travel time in hours")
]

# Create Output Parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Create Chain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Process User Input
if st.button("Get Travel Options"):
    if source and destination:
        query = prompt_template.format(source=source, destination=destination)
        
        try:
            # Run LLMChain to generate response
            raw_response = chain.run({"source": source, "destination": destination})

            # Attempt to parse structured response
            try:
                parsed_response = output_parser.parse(raw_response)
            except Exception:
                st.warning("Displaying output:")
                st.write(raw_response)
                st.stop()

            # Display Results
            st.subheader("Recommended Travel Options:")
            for option in parsed_response:
                st.write(f"**Mode:** {option['travel_mode']}")
                st.write(f"**Estimated Cost:** ${option['estimated_cost']}")
                st.write(f"**Duration:** {option['duration']} hours")
                st.write("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.warning("Please enter both source and destination.")
