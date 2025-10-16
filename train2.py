import pandas as pd
from sklearn.svm import SVC
import os
import pickle

# Load training data
train_df = pd.read_csv('data/train2.csv')
print("Train data loaded successfully, shape:", train_df.shape)

# Separate features and target
X = train_df.drop(columns=['species'])
y = train_df['species']

# Train model
model = SVC(random_state=42, max_iter=1000)
model.fit(X, y)
print("Model trained successfully")

# Save trained model
os.makedirs('models', exist_ok=True)
with open('models/model2.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully to models/model2.pkl")
