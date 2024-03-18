import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import scipy.stats as stats


# Function to train autoencoder
def train_autoencoder(data, hidden_neurons, hidden_layers):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data[['Transaction_Amount']])

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]

    autoencoder = models.Sequential()
    autoencoder.add(layers.Dense(hidden_neurons, activation='relu', input_shape=(input_dim,)))
    for _ in range(hidden_layers - 1):
        autoencoder.add(layers.Dense(hidden_neurons, activation='relu'))
    autoencoder.add(layers.Dense(input_dim, activation='sigmoid'))

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_train, X_train, epochs=20, batch_size=64, shuffle=True, validation_data=(X_test, X_test))

    return autoencoder, scaler


# Function to retrieve data from SQL database
def get_data_from_database(host, port, user, password, database):
    try:
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )

        query = "SELECT * FROM purchase_data"
        data = pd.read_sql(query, connection)
        connection.close()

        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# Function to save anomalies as CSV
def save_anomalies_csv(anomalies):
    anomalies.to_csv('anomalies.csv', index=False)


# Calculate total entropy change
def calculate_entropy_change(data, anomalies):
    normal_transactions = data[data.index.isin(anomalies.index) == False]
    anomaly_entropy = stats.entropy(anomalies['Transaction_Amount'])
    normal_entropy = stats.entropy(normal_transactions['Transaction_Amount'])
    entropy_change = normal_entropy - anomaly_entropy
    return entropy_change


# Streamlit app
def main():
    st.title("Anomaly Detection on a structured workload")
    st.markdown("---")

    # Sidebar for hyperparameter tuning
    st.sidebar.title("Hyperparameter Tuning")
    hidden_neurons = st.sidebar.slider("Number of Neurons in Hidden Layers", 32, 512, step=32, value=128)
    hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, step=1, value=2)

    st.sidebar.markdown("---")

    # Database connection settings
    st.sidebar.title("Database Connection")
    host = st.sidebar.text_input("Host", value="localhost", help="Enter the database host address")
    port = st.sidebar.text_input("Port", value="3306", help="Enter the database port number")
    user = st.sidebar.text_input("Username", value="root", help="Enter the database username")
    password = st.sidebar.text_input("Password", type="password", help="Enter the database password")
    database = st.sidebar.text_input("Database", value="Hackon", help="Enter the database name")

    st.sidebar.markdown("---")

    if st.sidebar.button("Connect to Database"):
        data = get_data_from_database(host, int(port), user, password, database)

        if data is not None:
            st.success("Connected to the database successfully.")
            st.write(data.head())

            st.write("*Training the Autoencoder...*")
            autoencoder, scaler = train_autoencoder(data, hidden_neurons, hidden_layers)
            st.write("*Autoencoder trained successfully!*")
            st.markdown("---")

            # Detect anomalies
            X = scaler.transform(data[['Transaction_Amount']])
            reconstructed_data = autoencoder.predict(X)
            reconstruction_errors = np.mean(np.abs(reconstructed_data - X), axis=1)
            threshold = np.percentile(reconstruction_errors, 95)  # Adjust percentile as needed
            anomalies = data[reconstruction_errors > threshold]

            # Save anomalies as CSV
            if not anomalies.empty:
                save_anomalies_csv(anomalies)
                st.success("Anomalies saved as 'anomalies.csv'")
            else:
                st.write("No anomalies detected.")

            # Display number of anomalies detected
            st.subheader("Number of Anomalies Detected:")
            st.write(len(anomalies))

            # Display explanation of entropy change and total entropy
            st.subheader("Understanding Entropy Change and Total Entropy")
            st.write(
                "In the context of anomaly detection, entropy change refers to the difference "
                "in the randomness or unpredictability of transaction amounts between normal "
                "transactions and anomalies. A higher entropy change indicates a greater deviation "
                "from the normal pattern of transactions. Total entropy represents the overall "
                "level of randomness in the transaction amounts across all data points."
            )

            # Calculate and display total entropy change
            total_entropy_change = calculate_entropy_change(data, anomalies)
            st.subheader("Total Entropy Change:")
            st.write(total_entropy_change)

            # Display anomalies detected
            if not anomalies.empty:
                st.subheader("Anomalies Detected:")
                st.write(anomalies)
                st.markdown("---")

            # Visualization of anomalies
            if not anomalies.empty:
                st.subheader("Visualizations of Anomalies")

                # Histogram
                st.write("**Histogram of Transaction Amount for Anomalies**")
                fig, ax = plt.subplots()
                sns.histplot(anomalies['Transaction_Amount'], bins=20, kde=True, color='skyblue', edgecolor='black')
                ax.set_xlabel('Transaction Amount')
                ax.set_ylabel('Frequency')
                st.pyplot(fig)

                # Time Series
                st.write("**Time Series of Transaction Amount for Anomalies**")
                fig, ax = plt.subplots()
                ax.plot(anomalies.index, anomalies['Transaction_Amount'], marker='o', linestyle='-')
                ax.set_xlabel('Time')
                ax.set_ylabel('Transaction Amount')
                st.pyplot(fig)

                # Box Plot
                st.write("**Box Plot of Transaction Amount for Anomalies**")
                fig, ax = plt.subplots()
                sns.boxplot(y=anomalies['Transaction_Amount'], color='salmon')
                ax.set_ylabel('Transaction Amount')
                st.pyplot(fig)

                # Density Plot
                st.write("**Density Plot of Transaction Amount for Anomalies**")
                fig, ax = plt.subplots()
                sns.kdeplot(data=anomalies['Transaction_Amount'], shade=True, color='orange')
                ax.set_xlabel('Transaction Amount')
                ax.set_ylabel('Density')
                st.pyplot(fig)


if __name__ == "__main__":
    main()
