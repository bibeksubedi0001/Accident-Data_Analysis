# Comprehensive Accident Data Analysis for Kathmandu Valley

## Project Overview
This project performs a detailed analysis of accident data in the Kathmandu Valley. It identifies accident hotspots, explores temporal trends, analyzes accident severity distribution, and provides predictive modeling for accident severity. Additionally, geographical visualization and correlation analysis reveal underlying patterns in the data. The project saves graphical outputs and a hotspot analysis report for further reference.

---

## Dataset Details
The dataset used in this analysis includes the following columns:
- **Accident_ID**: Unique identifier for each accident.
- **Location**: Location of the accident (e.g., Maitighar, Kalanki, Koteshwor).
- **Date**: Date of the accident in `YYYY-MM-DD` format.
- **Time**: Time of the accident in `HH:MM` format.
- **Severity**: Accident severity (Low, Medium, High).
- **Latitude**: Geographical latitude of the accident location.
- **Longitude**: Geographical longitude of the accident location.

---

## Analysis Features
### 1. Data Exploration
- Displays the first few rows, dataset structure, and a summary of numerical columns.
- Identifies and handles missing values.

### 2. Accident Hotspot Analysis
- Groups data by location to identify top accident-prone areas.
- Visualizes the results using a bar plot.

### 3. Time-Based Accident Analysis
- Extracts the hour from the `Time` column.
- Analyzes hourly accident distribution using a line plot.

### 4. Severity Distribution
- Displays the frequency of accidents categorized by severity (Low, Medium, High).
- Visualizes the results using a bar plot.

### 5. Monthly Trends
- Extracts the month from the `Date` column.
- Analyzes monthly accident trends using a bar plot.

### 6. Correlation Analysis
- Encodes `Severity` into numeric values (Low=1, Medium=2, High=3).
- Computes a correlation matrix and visualizes it using a heatmap.

### 7. Predictive Modeling
- Implements a Random Forest Classifier to predict accident severity based on time and location.
- Evaluates the model using a classification report and accuracy score.

### 8. Geographical Visualization
- Maps accident locations on a scatter plot, color-coded by severity level.

---

## Outputs
1. **Graphical Figures**:
   - **Hotspots.png**: Top accident hotspots.
   - **Hourly_Accidents.png**: Hourly accident distribution.
   - **Accident Severity.png**: Severity distribution.
   - **Monthwise_Trend.png**: Monthly accident trends.
   - **Correlations.png**: Heatmap of variable correlations.
   - **Geographical Visualization.png**: Map of accident locations.

2. **Hotspot Analysis Report**:
   - File: `Hotspot_Analysis.csv`
   - Contains the list of accident locations and their corresponding accident counts.

3. **Predictive Modeling Results**:
   - Classification Report:
     ```
     Precision: 0.26
     Recall: 0.27
     F1-Score: 0.25
     Accuracy: 27%
     ```

---

## Dependencies
The following Python libraries are required:
- **pandas**: Data manipulation and analysis.
- **matplotlib**: Static plotting for visualizations.
- **seaborn**: Enhanced data visualizations.
- **numpy**: Numerical operations.
- **scikit-learn**: Machine learning algorithms and evaluation metrics.
- **os**: Directory creation and file handling.

---

## Installation
1. Clone the repository or download the project files.
2. Install the required dependencies:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn numpy
   ```
3. Place the dataset `Kathmandu_Accident_Data.csv` in the project directory.


---

## Summary of Findings
- **Top Accident Locations**:
  1. Maitighar: 18 accidents
  2. Koteshwor: 14 accidents
  3. Kalanki: 13 accidents
  4. Durbar Marg: 12 accidents
  5. Thamel: 12 accidents

- **Accident Severity Distribution**:
  - High: 38 accidents
  - Low: 36 accidents
  - Medium: 26 accidents

- **Model Accuracy**:
  - The predictive model achieved an accuracy score of **27%**, indicating limited predictive power due to the dataset size and feature availability.

---

## Project Structure
```
Accident Data Analysis/
│
├── analysis.py         # Main Python script
├── Kathmandu_Accident_Data.csv  # Input dataset
├── Hotspot_Analysis.csv         # Exported hotspot report
├── Figures/                     # Saved graphical outputs
│   ├── Hotspots.png
│   ├── Hourly_Accidents.png
│   ├── Accident Severity.png
│   ├── Monthwise_Trend.png
│   ├── Correlations.png
│   ├── Geographical Visualization.png
```

---

## Limitations
1. **Data Size**: The model's accuracy could improve with a larger dataset.
2. **Feature Availability**: Including weather and traffic data might enhance analysis and predictive modeling.
3. **Predictive Model Performance**: Low accuracy suggests that further feature engineering or alternative models may be needed.

---

