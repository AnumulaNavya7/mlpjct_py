# STEP 1: Upload CSV
from google.colab import files
uploaded = files.upload()

# STEP 2: Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# STEP 3: Load Data
df = pd.read_csv('survey.csv')
print("✅ Data loaded. Shape:", df.shape)

# STEP 4: Drop unnecessary columns
df.drop(columns=[col for col in ['Timestamp', 'state', 'comments'] if col in df.columns], inplace=True)

# STEP 5: Fill missing values
df.fillna('Unknown', inplace=True)

# STEP 6: Encode categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# STEP 7: Train-Test Split
X = df.drop('treatment', axis=1)
y = df['treatment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 8: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 9: Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Class Distribution
sns.countplot(x=y)
plt.title("Treatment Class Distribution")
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[sorted_idx], y=features[sorted_idx])
plt.title("🔍 Feature Importance")
plt.show()

# STEP 10: Predict for a new person
print("\n🧠 ENTER DETAILS TO PREDICT MENTAL HEALTH RISK:\n")

# Input more questions to improve accuracy
user_data = {
    'Age': int(input("Enter Age: ")),
    'Gender': input("Gender (Male/Female/Other): "),
    'Country': input("Country: "),
    'self_employed': input("Are you self-employed? (Yes/No): "),
    'family_history': input("Any family history of mental illness? (Yes/No): "),
    'work_interfere': input("How often does mental health interfere with your work? (Often/Sometimes/Rarely/Never): "),
    'no_employees': input("Company size (6-25, 26-100, 100-500, More than 500): "),
    'remote_work': input("Do you work remotely? (Yes/No): "),
    'tech_company': input("Is it a tech company? (Yes/No): "),
    'benefits': input("Mental health benefits from employer? (Yes/No): "),
    'care_options': input("Do you know the care options available? (Yes/No/Not sure): "),
    'wellness_program': input("Any wellness program at work? (Yes/No/Don't know): "),
    'seek_help': input("Does your company encourage seeking help? (Yes/No): "),
    'anonymity': input("Is anonymity provided for mental health? (Yes/No): "),
    'leave': input("How easy is it to take mental health leave? (Very easy/Somewhat easy/Somewhat difficult/Very difficult): "),
    'mental_health_consequence': input("Consequence of discussing mental health at work? (Yes/No/Maybe): "),
    'phys_health_consequence': input("Consequence of discussing physical health at work? (Yes/No/Maybe): "),
    'coworkers': input("Can you talk to coworkers about mental health? (Yes/No/Some of them): "),
    'supervisor': input("Can you talk to supervisor? (Yes/No/Some of them): "),
    'mental_health_interview': input("Would you discuss mental health in interview? (Yes/No/Maybe): "),
    'phys_health_interview': input("Would you discuss physical health in interview? (Yes/No/Maybe): "),
    'mental_vs_physical': input("Is mental health taken as seriously as physical health? (Yes/No/Don't know): "),
    'obs_consequence': input("Negative consequence from discussing at work? (Yes/No): "),
}

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# Encode input data
for col in input_df.columns:
    if col in label_encoders:
        le = label_encoders[col]
        if input_df[col][0] in le.classes_:
            input_df[col] = le.transform([input_df[col][0]])
        else:
            input_df[col] = -1  # Unknown/new label

# Predict
prediction = model.predict(input_df)[0]
treatment_needed = label_encoders['treatment'].inverse_transform([prediction])[0]

print(f"\n🔍 Prediction: This person is likely to {'need' if treatment_needed == 'Yes' else 'not need'} mental health treatment.")