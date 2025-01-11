import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# Step 1: Load the dataset
file_path = "Kathmandu_Accident_Data.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)
save_folder = "D:\\My Projects\\Accident Data Analysis\\"
os.makedirs(save_folder, exist_ok=True)
# Step 2: Explore the dataset
print("First few rows of the dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nSummary of numerical columns:")
print(data.describe())

# Step 3: Data Cleaning
print("\nChecking for missing values:")
print(data.isnull().sum())

# Handle missing values
data_cleaned = data.dropna()
print("\nData after cleaning:")
print(data_cleaned.info())

# Step 4: Accident Hotspot Analysis
hotspots = data_cleaned.groupby('Location')['Accident_ID'].count().reset_index()
hotspots = hotspots.rename(columns={"Accident_ID": "Accident_Count"})
hotspots = hotspots.sort_values(by="Accident_Count", ascending=False)

print("\nTop 5 Accident-Prone Locations:")
print(hotspots.head())

# Plot Accident Hotspots
plt.figure(figsize=(12, 6))
sns.barplot(x='Location', y='Accident_Count', data=hotspots, palette="Blues_d")
plt.title('Top Accident Hotspots in Kathmandu Valley')
plt.xlabel('Location')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'Hotspots.png'))
plt.show()

# Step 5: Time-Based Analysis
data_cleaned['Hour'] = pd.to_datetime(data_cleaned['Time'], format='%H:%M').dt.hour
hourly_accidents = data_cleaned.groupby('Hour')['Accident_ID'].count().reset_index()
hourly_accidents = hourly_accidents.rename(columns={"Accident_ID": "Accident_Count"})

plt.figure(figsize=(12, 6))
sns.lineplot(x='Hour', y='Accident_Count', data=hourly_accidents, marker='o', color='green')
plt.title('Hourly Accident Distribution')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'Hourly_Accidents.png'))
plt.show()

# Step 6: Severity-Based Analysis
severity_counts = data_cleaned['Severity'].value_counts()

plt.figure(figsize=(8, 6))
sns.barplot(x=severity_counts.index, y=severity_counts.values, palette="viridis")
plt.title('Accident Severity Distribution')
plt.xlabel('Severity')
plt.ylabel('Number of Accidents')
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'Accident Severity.png'))
plt.show()

# Step 7: Monthly Trends
data_cleaned['Month'] = pd.to_datetime(data_cleaned['Date']).dt.month
monthly_accidents = data_cleaned.groupby('Month')['Accident_ID'].count().reset_index()
monthly_accidents = monthly_accidents.rename(columns={"Accident_ID": "Accident_Count"})

plt.figure(figsize=(12, 6))
sns.barplot(x='Month', y='Accident_Count', data=monthly_accidents, palette="coolwarm")
plt.title('Accidents by Month')
plt.xlabel('Month')
plt.ylabel('Number of Accidents')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'Monthwise_Trend.png'))
plt.show()


# Step 8: Correlation Analysis
# Encode Severity to numerical values
severity_mapping = {"Low": 1, "Medium": 2, "High": 3}
data_cleaned['Severity_Encoded'] = data_cleaned['Severity'].map(severity_mapping)

correlation_matrix = data_cleaned[['Hour', 'Severity_Encoded', 'Latitude', 'Longitude']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Between Variables")
plt.savefig(os.path.join(save_folder, 'Correlations.png'))
plt.show()

# Step 9: Predictive Modeling (Optional)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Prepare data for prediction
X = data_cleaned[['Hour', 'Latitude', 'Longitude']]
y = data_cleaned['Severity_Encoded']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Step 10: Geographical Visualization
if 'Latitude' in data_cleaned.columns and 'Longitude' in data_cleaned.columns:
    plt.figure(figsize=(10, 8))
    plt.scatter(data_cleaned['Longitude'], data_cleaned['Latitude'], c=data_cleaned['Severity_Encoded'], cmap='Set1', alpha=0.5)
    plt.colorbar(label='Severity Level (Low=1, High=3)')
    plt.title('Geographical Accident Distribution')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'Geographical Visualization.png'))
    plt.show()

# Step 11: Insights and Reporting
hotspots.to_csv('Hotspot_Analysis.csv', index=False)
print("\nHotspot analysis exported to 'Hotspot_Analysis.csv'.")
print("\nSummary of Findings:")
print(f"Total Number of Accidents: {len(data_cleaned)}")
print("Top Accident Locations:")
print(hotspots.head())
print("\nAccident Severity Distribution:")
print(severity_counts)