from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

# Load the trained model with custom objects
model = load_model('sales_funnel_optimization_dqn_model.h5', custom_objects={'mse': MeanSquaredError()})

# Define a sample state for a lead
engagement_score = 0.45  # Adjusted to match the normalized scale (0 to 1)
funnel_stage = 1         # "Warm" stage (0 for Cold, 1 for Warm, 2 for Hot)

# Create the state array for the lead (normalized engagement score)
test_state = np.array([[engagement_score, funnel_stage]])

# Use the model to predict Q-values for each action
q_values = model.predict(test_state)

# Map actions to labels
action_map = {
    0: 'Send an Educational Email',
    1: 'Invite to a Webinar or Demo',
    2: 'Set Up a Call with a Specialist'
}
recommended_action = action_map[np.argmax(q_values)]

# Output the results
print("Predicted Q-values for actions:", q_values)
print("Recommended action:", recommended_action)
