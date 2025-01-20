import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Caching model and data loading
@st.cache_resource
def load_model_cached():
    try:
        model = load_model('sales_funnel_optimization_dqn_model.h5', custom_objects={'mse': MeanSquaredError()})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("updated_lead_data_with_feedback_loop.csv")
        
        # Handle missing values silently
        missing_values = df.isnull().sum()
        if missing_values.any():
            df = df.dropna()  # Drop rows with NaN values or use fillna(0) to fill with zeros

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load model and data
model = load_model_cached()
df = load_data()

# Check if model and data loaded successfully
if model is None or df is None:
    st.stop()

# Define funnel stage mapping for model predictions
funnel_stage_map = {"Cold": 0, "Warm": 1, "Hot": 2}
df['FunnelStageMapped'] = df['FunnelStage'].map(funnel_stage_map)

# Generate predictions for each row in the dataset
states = df[['EngagementScore', 'FunnelStageMapped']].values
q_values = model.predict(states)
df['predicted_action'] = np.argmax(q_values, axis=1)

# Define action map and add predicted actions to DataFrame
action_map = {
    0: 'Send an Educational Email',
    1: 'Invite to a Webinar or Demo',
    2: 'Set Up a Call with a Specialist'
}
df['predicted_conversion'] = df['predicted_action'].map(action_map)

st.title("Sales Funnel Optimization Dashboard")

# Add description section in the sidebar
st.sidebar.header("Project Overview")
st.sidebar.write("""
This Sales Funnel Optimization Dashboard leverages a machine learning model to recommend actions that can help improve lead engagement across different stages of the sales funnel. The dashboard provides insights into funnel performance and suggests optimal actions to move leads closer to conversion.
""")

st.sidebar.subheader("Key Terms")
st.sidebar.markdown("**Engagement Score**: A numerical value representing the level of interaction a lead has with the brand. Higher scores indicate stronger engagement, while lower scores suggest minimal interest.")
st.sidebar.markdown("**Funnel Stage**: The position of a lead in the buying journey, typically segmented as Cold, Warm, or Hot.")
st.sidebar.markdown("**Recommended Actions**:")
st.sidebar.markdown("""
- **Send an Educational Email**: Provides valuable content to keep leads engaged and informed.
- **Invite to a Webinar or Demo**: Encourages leads to participate in live sessions, increasing their interest level.
- **Set Up a Call with a Specialist**: Engages high-interest leads directly to discuss products or services.
""")

# Define tabs
tab1, tab2 = st.tabs(["Predict", "Insights"])

# Tab 1: Prediction Interface
with tab1:
    st.header("Predict Sales Funnel Action")
    engagement_score = st.slider("Engagement Score", 0, 100, 45)
    funnel_stage = st.selectbox("Funnel Stage", ["Cold", "Warm", "Hot"])

    # Map funnel stage to integer for prediction
    funnel_stage = funnel_stage_map[funnel_stage]

    if st.button("Submit"):
        # Normalize engagement score for prediction
        test_state = np.array([[engagement_score / 100, funnel_stage]])
        q_values = model.predict(test_state)

        recommended_action = action_map[np.argmax(q_values)]

        st.write("Predicted Q-values for actions:", q_values)
        st.write("Recommended action:", recommended_action)

# Tab 2: Insights and Dashboard
with tab2:
    st.header("Funnel Performance and Model Health Dashboard")
    st.markdown("This dashboard provides insights into funnel performance and assesses model health.")

    # Funnel Performance Visualizations

    # Visualization 1: Conversion Rate across Funnel Stages
    st.subheader("1. Conversion Rate by Funnel Stage")
    funnel_conversion = df.groupby('FunnelStage')['Conversion (Target)'].mean().dropna().reset_index()
    fig = px.bar(funnel_conversion, x='FunnelStage', y='Conversion (Target)', color='FunnelStage', title="Conversion Rate by Funnel Stage")
    fig.update_layout(xaxis_title="Funnel Stage", yaxis_title="Conversion Rate")
    st.plotly_chart(fig)

    # Visualization 2: Lead Status Distribution in Funnel
    st.subheader("2. Lead Status Distribution in Funnel")
    funnel_status = df.groupby(['FunnelStage', 'LeadStatus']).size().unstack().fillna(0)
    fig, ax = plt.subplots()
    funnel_status.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_xlabel("Funnel Stage")
    ax.set_ylabel("Number of Leads")
    st.pyplot(fig)

    # Visualization 3: Conversion Rate by Engagement Score
    st.subheader("3. Conversion Rate by Engagement Score")
    st.caption("Conversion rates for different engagement score ranges.")

    # Use pd.cut to create bins and then convert them to strings
    df['EngagementScoreGroup'] = pd.cut(df['EngagementScore'], bins=5).astype(str)
    engagement_conversion = df.groupby('EngagementScoreGroup')['Conversion (Target)'].mean().reset_index()

    # Plotting with Plotly
    fig = px.bar(engagement_conversion, x='EngagementScoreGroup', y='Conversion (Target)', title="Conversion Rate by Engagement Score")
    fig.update_layout(xaxis_title="Engagement Score Group", yaxis_title="Conversion Rate")
    st.plotly_chart(fig)

    # Visualization 4: Device Type Impact on Conversion Rate
    st.subheader("4. Device Type Impact on Conversion Rate")
    device_conversion = df.groupby('DeviceType')['Conversion (Target)'].mean().reset_index()
    fig = px.bar(device_conversion, x='DeviceType', y='Conversion (Target)', title="Conversion Rate by Device Type")
    fig.update_layout(xaxis_title="Device Type", yaxis_title="Conversion Rate")
    st.plotly_chart(fig)

    # Model Health Section

    st.header("Model Health Overview")

    # Visualization 5: Confusion Matrix for Conversion Prediction
    st.subheader("5. Confusion Matrix for Model Predictions")
    y_true = df['Conversion (Target)']
    y_pred = df['predicted_action']  # Ensure 'predicted_action' exists in your dataset
    conf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    # Model Health Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    st.subheader("Model Performance Metrics")
    st.markdown(f"**Accuracy**: {accuracy:.2f}")
    st.markdown(f"**Precision**: {precision:.2f}")
    st.markdown(f"**Recall**: {recall:.2f}")

    # Visualization 6: Feature Importance (Placeholder)
    st.subheader("6. Feature Importance in Conversion Prediction")
    feature_importance = {'EngagementScore': 0.3, 'InterestLevel': 0.25, 'TimeSpent (minutes)': 0.15, 'TotalEmails': 0.1, 'PagesViewed': 0.1, 'FollowUpEmails': 0.05, 'ResponseTime (hours)': 0.05}
    feature_importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title="Feature Importance for Conversion Prediction")
    st.plotly_chart(fig)

    # Visualization 7: Precision and Recall by Funnel Stage
    st.subheader("7. Precision and Recall by Funnel Stage")
    stage_metrics = []
    for stage in df['FunnelStage'].unique():
        stage_data = df[df['FunnelStage'] == stage]
        if len(stage_data) > 0:
            stage_y_true = stage_data['Conversion (Target)']
            stage_y_pred = stage_data['predicted_action']
            stage_precision = precision_score(stage_y_true, stage_y_pred, average='weighted')
            stage_recall = recall_score(stage_y_true, stage_y_pred, average='weighted')
            stage_metrics.append({'FunnelStage': stage, 'Precision': stage_precision, 'Recall': stage_recall})
    stage_metrics_df = pd.DataFrame(stage_metrics)