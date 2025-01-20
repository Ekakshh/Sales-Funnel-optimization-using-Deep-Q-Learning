import numpy as np
import pandas as pd
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess the dataset
data_path = 'updated_lead_data_with_feedback_loop.csv'
lead_data = pd.read_csv(data_path)

# Preprocessing Steps

# Handle missing values
lead_data['EngagementScore'].fillna(lead_data['EngagementScore'].mean(), inplace=True)
lead_data.dropna(subset=['FunnelStage', 'LeadID'], inplace=True)

# Convert funnel stage to numeric values
funnel_stage_map = {'Cold': 0, 'Warm': 1, 'Hot': 2}
lead_data['FunnelStage'] = lead_data['FunnelStage'].map(funnel_stage_map)

# Normalize the EngagementScore to a range between 0 and 1
lead_data['EngagementScore'] = lead_data['EngagementScore'] / 100  # Assuming max score is 100

# Confirm preprocessing by viewing a sample of the data
print(lead_data.head())

# Define environment settings
states = ['EngagementScore', 'FunnelStage']  # State features
actions = ['Send an Educational Email', 'Invite to a Webinar or Demo', 'Set Up a Call with a Specialist']  # Updated actions
state_size = len(states)
action_size = len(actions)

# Hyperparameters
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = deque(maxlen=2000)

# Deep Q-Network
def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

model = build_model()

# Function to get the current state of a lead
def get_state(lead_id):
    lead = lead_data[lead_data['LeadID'] == lead_id]
    return np.array([lead['EngagementScore'].values[0], lead['FunnelStage'].values[0]])

# Function to choose an action
def choose_action(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # Explore
    q_values = model.predict(state)
    return np.argmax(q_values[0])  # Exploit

# Function to execute an action and return reward
def execute_action(lead_id, action):
    # Get the engagement score before action
    current_score = lead_data.loc[lead_data['LeadID'] == lead_id, 'EngagementScore'].values[0]
    
    # Apply the action - adjust score based on action type
    if action == 0:  # Send an Educational Email
        score_increase = random.uniform(0.01, 0.03)  # Small increase for initial engagement
    elif action == 1:  # Invite to a Webinar or Demo
        score_increase = random.uniform(0.03, 0.05)  # Moderate increase for further engagement
    elif action == 2:  # Set Up a Call with a Specialist
        score_increase = random.uniform(0.05, 0.07)  # Higher increase for committed leads
    
    # Update the engagement score
    lead_data.loc[lead_data['LeadID'] == lead_id, 'EngagementScore'] += score_increase
    
    # Calculate reward based on score improvement
    new_score = lead_data.loc[lead_data['LeadID'] == lead_id, 'EngagementScore'].values[0]
    reward = new_score - current_score
    done = new_score >= 1.0  # Define a high engagement score as terminal state
    
    return reward, done

# Store experience in memory
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Training the DQN model
def replay():
    global epsilon
    if len(memory) < batch_size:
        return
    
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Main training loop
episodes = 100
for episode in range(episodes):
    lead_id = random.choice(lead_data['LeadID'].tolist())
    state = get_state(lead_id).reshape(1, state_size)
    
    for time in range(500):
        action = choose_action(state)
        reward, done = execute_action(lead_id, action)
        next_state = get_state(lead_id).reshape(1, state_size)
        store_experience(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            print(f"Episode: {episode}/{episodes}, Time: {time}, Reward: {reward}, Epsilon: {epsilon:.2}")
            break
        
    replay()

# Save the trained model for future predictions
model.save('sales_funnel_optimization_dqn_model.h5')