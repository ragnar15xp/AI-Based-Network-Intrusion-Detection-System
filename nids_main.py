import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")

if "model" not in st.session_state:
    st.session_state.model = None

if "trained" not in st.session_state:
    st.session_state.trained = False

def generate_network_data(samples=6000):
    np.random.seed(42)
    data = {
        "Destination_Port": np.random.randint(20, 65535, samples),
        "Flow_Duration": np.random.randint(1, 100000, samples),
        "Total_Fwd_Packets": np.random.randint(1, 300, samples),
        "Total_Bwd_Packets": np.random.randint(1, 200, samples),
        "Packet_Length_Mean": np.random.uniform(20, 1500, samples),
        "Flow_IAT_Mean": np.random.uniform(1, 5000, samples),
        "Active_Mean": np.random.uniform(0, 2000, samples),
        "Label": np.random.choice([0, 1], samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    attack = df["Label"] == 1
    df.loc[attack, "Total_Fwd_Packets"] += np.random.randint(100, 400, attack.sum())
    df.loc[attack, "Flow_Duration"] = np.random.randint(1, 500, attack.sum())
    df.loc[attack, "Packet_Length_Mean"] += np.random.randint(200, 800, attack.sum())
    return df

st.sidebar.title("Control Panel")

train_size = st.sidebar.slider("Training Data Percentage", 60, 90, 80)
trees = st.sidebar.slider("Number of Trees", 50, 300, 150)

df = generate_network_data()

X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=(100 - train_size) / 100, random_state=42
)

if st.button("Train Model Now"):
    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    st.session_state.model = model
    st.session_state.trained = True
    st.success("Model trained successfully")

if st.session_state.trained:
    model = st.session_state.model
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc * 100:.2f}%")
    c2.metric("Total Samples", len(df))
    c3.metric("Detected Attacks", np.sum(y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.text(classification_report(y_test, y_pred))

flow_duration = st.number_input("Flow Duration", 0, 100000, 400)
fwd_packets = st.number_input("Total Forward Packets", 0, 600, 120)
bwd_packets = st.number_input("Total Backward Packets", 0, 400, 80)
pkt_length = st.number_input("Packet Length Mean", 0, 1500, 600)
iat_mean = st.number_input("Flow IAT Mean", 0, 5000, 300)
active_mean = st.number_input("Active Mean", 0, 2000, 200)

if st.button("Analyze Traffic"):
    if not st.session_state.trained:
        st.error("Train the model first")
    else:
        model = st.session_state.model
        test_data = np.array([[80, flow_duration, fwd_packets, bwd_packets,
                               pkt_length, iat_mean, active_mean]])
        pred = model.predict(test_data)[0]
        if pred == 1:
            st.error("INTRUSION DETECTED")
        else:
            st.success("NORMAL TRAFFIC")
