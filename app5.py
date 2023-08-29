import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize the session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Function to perform classification and return accuracy and confusion matrix
def classify(X_train, y_train, X_test, y_test):
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    return accuracy, conf_mat

# Main App
st.title("📊 Simple Data Analysis App")
st.write("Upload your data, explore it, and model it.")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Upload", "Data Exploration", "Model Building"])

# Data Upload
if section == "Data Upload":
    st.header("📤 Data Upload")
    data_file = st.file_uploader("Upload your dataset (.csv only)", type=["csv"])

    if data_file is not None:
        df = pd.read_csv(data_file)
        st.session_state.df = df  # Save to session state
        st.success("Data uploaded successfully!")
        st.write("### Preview")
        st.dataframe(df.head())

# Data Exploration
elif section == "Data Exploration":
    st.header("🔍 Data Exploration")
    if st.session_state.df is not None:
        df = st.session_state.df  # Retrieve from session state
        st.write("### Data Types")
        st.write(df.dtypes)

        # Show selected columns
        cols = st.multiselect("Choose columns", df.columns.tolist())
        st.dataframe(df[cols].head())

        # Show statistics
        st.write("### Statistics")
        st.write(df.describe())

        # Plot
        st.write("### Plot")
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        x_axis = st.selectbox("Choose x-axis", columns)
        y_axis = st.selectbox("Choose y-axis", columns)
        fig, ax = plt.subplots(figsize=(12, 6))  # Create a figure and axis object
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)  # Use the axis object
        st.pyplot(fig)  # Pass the figure object explicitly
        
    else:
        st.warning("No data uploaded yet. Go to the Data Upload section.")

# Model Building
elif section == "Model Building":
    st.header("🛠 Model Building")
    if st.session_state.df is not None:
        df = st.session_state.df  # Retrieve from session state
        target = st.selectbox("Select target variable", df.columns.tolist())
        st.write(f"Target: {target}")

        if st.button("Train Model"):
            progress = st.progress(0)
            st.write("Training in progress...")
            X = df.drop(target, axis=1)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            accuracy, conf_mat = classify(X_train, y_train, X_test, y_test)
            progress.progress(100)
            st.write(f"Accuracy: {accuracy}")
            st.write("Confusion Matrix:")
            st.write(conf_mat)

    else:
        st.warning("No data uploaded yet. Go to the Data Upload section.")
