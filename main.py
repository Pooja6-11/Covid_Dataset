import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
@st.cache_data  # Use st.cache_data instead of st.cache
def load_data():
    data = pd.read_csv('Covid Data.csv')  # Replace 'Covid Data.csv' with your dataset filename
    return data

# Function to train your machine learning model
def train_model(data):
    # Drop any non-numeric columns or columns that can't be converted to numeric
    data_numeric = data.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    
    # Separate features and target
    X = data_numeric.drop(columns=['SEX', 'PATIENT_TYPE', 'INTUBED', 'PNEUMONIA', 'PREGNANT', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
                       'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO', 'ICU'])  # Replace 'CLASIFFICATION_FINAL' with your target column name
    y = data_numeric['CLASIFFICATION_FINAL']  # Replace 'CLASIFFICATION_FINAL' with your target column name
    
    # Train the model (removed for brevity)
    
    return None  # Return None as we are not training the model for this example

# Function to plot visualizations
def plot_visualizations(data, target_column):
    st.subheader(f'Data Distribution for {target_column}')
    st.write(data[target_column].value_counts())

    st.subheader('Histogram of Age by Classification')
    fig, ax = plt.subplots()  # Create a new figure
    sns.histplot(data, x='AGE', hue=target_column, multiple='stack',palette='husl', ax=ax)
    st.pyplot(fig)  # Pass the figure to st.pyplot()

    st.subheader('Count of Gender by Classification')
    fig, ax = plt.subplots()  # Create a new figure
    sns.countplot(data=data, x='SEX', hue=target_column,palette='Set2', ax=ax)
    st.pyplot(fig)  # Pass the figure to st.pyplot()

# Main function to run the Streamlit app
def main():
    st.title('COVID-19 Data Analysis')

    # Load the dataset
    data = load_data()

    # Display a preview of the dataset
    st.sidebar.header('Dashboard')
    st.sidebar.subheader('Dataset Preview')
    st.sidebar.write(data.head())

    # Train the machine learning model (removed for brevity)

    # Add some user-friendly elements
    #st.sidebar.subheader('Interactive Features')

    # Sliders for numerical columns
    st.sidebar.subheader('Filter Data')

    age_values = sorted(data['AGE'].unique())
    selected_age_range = st.sidebar.selectbox('Select Age Range', age_values)

    filtered_data = data[data['AGE'] == selected_age_range]

    # Show visualizations based on selected age range
    plot_visualizations(filtered_data, 'CLASIFFICATION_FINAL')
    # Filter data based on classification
    classification_values = data['CLASIFFICATION_FINAL'].unique()
    selected_classification = st.sidebar.selectbox('Select Classification', classification_values)

    filtered_data = data[data['CLASIFFICATION_FINAL'] == selected_classification]

    # Show visualizations based on selected classification
    plot_visualizations(filtered_data, 'CLASIFFICATION_FINAL')

if __name__ == '__main__':
    main()
