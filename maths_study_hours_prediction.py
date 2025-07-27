import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== Student Study Hours Prediction Project ===")
print("Predicting study hours needed to achieve target exam scores")
print()

# 1. Load the dataset
print("Loading UCI Student Performance Dataset...")
# You can download from: https://archive.ics.uci.edu/ml/datasets/student+performance
# Using math dataset (student-mat.csv)
try:
    # Update this path to your dataset location
    data = pd.read_csv('C:/Users/OSL/Downloads/student+performance (1)/student/student-mat.csv', sep=';')
    print(f"Dataset loaded successfully! Shape: {data.shape}")
except FileNotFoundError:
    print("Error: 'student-mat.csv' not found.")
    print("Please download from UCI repository and place in the same directory.")
    print("URL: https://archive.ics.uci.edu/ml/datasets/student+performance")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Data Exploration
print("\n=== Dataset Overview ===")
print("First 5 rows:")
print(data.head())
print(f"\nDataset shape: {data.shape}")
print(f"\nColumn names: {list(data.columns)}")

# Check for missing values
print(f"\nMissing values:\n{data.isnull().sum().sum()} total missing values")

# Display basic statistics
print(f"\nGrade distributions:")
print(f"G1 (1st period): {data['G1'].min()}-{data['G1'].max()}, mean: {data['G1'].mean():.1f}")
print(f"G2 (2nd period): {data['G2'].min()}-{data['G2'].max()}, mean: {data['G2'].mean():.1f}")
print(f"G3 (final): {data['G3'].min()}-{data['G3'].max()}, mean: {data['G3'].mean():.1f}")
print(f"Study time distribution: {data['studytime'].value_counts().sort_index().to_dict()}")

# 3. Reverse Engineering: Create Study Hours Target
print("\n=== Reverse Engineering Process ===")
print("Converting categorical study time to actual hours...")

# Map study time categories to actual hours (taking midpoint of ranges)
studytime_mapping = {
    1: 1.0,    # <2 hours -> 1 hour
    2: 3.5,    # 2-5 hours -> 3.5 hours  
    3: 7.5,    # 5-10 hours -> 7.5 hours
    4: 12.0    # >10 hours -> 12 hours
}

# Create our target variable: study_hours_needed
data['study_hours_needed'] = data['studytime'].map(studytime_mapping)

# Add some realistic variation based on other factors
np.random.seed(42)
# Students with more failures might need more study time
failure_adjustment = data['failures'] * 0.5
# Students with family support might need slightly less time
family_adjustment = np.where(data['famsup'] == 'yes', -0.3, 0.2)
# Students with more absences might need more focused study time
absence_adjustment = (data['absences'] / 20) * 0.5

# Apply adjustments
data['study_hours_needed'] = data['study_hours_needed'] + failure_adjustment + family_adjustment + absence_adjustment
# Ensure realistic bounds (0.5 to 15 hours)
data['study_hours_needed'] = np.clip(data['study_hours_needed'], 0.5, 15.0)

print(f"Study hours needed - Range: {data['study_hours_needed'].min():.1f} to {data['study_hours_needed'].max():.1f}")
print(f"Study hours needed - Mean: {data['study_hours_needed'].mean():.1f}")

# 4. Feature Selection and Engineering
print("\n=== Feature Engineering ===")

# Select relevant features for predicting study hours
selected_features = [
    'G1', 'G2',  # Previous grades (key predictors)
    'failures', 'absences',  # Academic history
    'age', 'Medu', 'Fedu',  # Demographics and parent education
    'studytime',  # Previous study habits
    'schoolsup', 'famsup', 'paid',  # Support systems
    'activities', 'nursery', 'higher',  # Educational background
    'internet', 'romantic',  # Lifestyle factors
    'freetime', 'goout', 'Dalc', 'Walc', 'health'  # Social and health factors
]

# Check if all features exist
available_features = [f for f in selected_features if f in data.columns]
print(f"Using {len(available_features)} features: {available_features}")

# Create feature matrix
X = data[available_features].copy()
y = data['study_hours_needed'].copy()

# Handle categorical variables
categorical_features = X.select_dtypes(include=['object']).columns
print(f"Categorical features to encode: {list(categorical_features)}")

# Label encode categorical features
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature])
    label_encoders[feature] = le

print(f"Final feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# 5. Data Visualization
print("\n=== Creating Visualizations ===")

# Create correlation matrix for key features
plt.figure(figsize=(12, 8))
key_features = ['G1', 'G2', 'failures', 'absences', 'age', 'studytime', 'study_hours_needed']
corr_matrix = data[key_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix: Key Features vs Study Hours Needed')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Distribution of study hours needed
plt.figure(figsize=(10, 6))
plt.hist(y, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Study Hours Needed')
plt.ylabel('Frequency')
plt.title('Distribution of Study Hours Needed')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('study_hours_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Study hours vs grades scatter plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.scatter(data['G1'], y, alpha=0.6, color='blue')
ax1.set_xlabel('G1 (1st Period Grade)')
ax1.set_ylabel('Study Hours Needed')
ax1.set_title('Study Hours vs 1st Period Grade')
ax1.grid(alpha=0.3)

ax2.scatter(data['G2'], y, alpha=0.6, color='green')
ax2.set_xlabel('G2 (2nd Period Grade)')
ax2.set_ylabel('Study Hours Needed')
ax2.set_title('Study Hours vs 2nd Period Grade')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('grades_vs_study_hours.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved: correlation_matrix.png, study_hours_distribution.png, grades_vs_study_hours.png")

# 6. Data Splitting and Preprocessing
print("\n=== Data Preprocessing ===")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")

# 7. Model Training
print("\n=== Model Training ===")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

# Train models and store results
model_results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    
    model_results[name] = {
        'MSE': mse,
        'R2': r2,
        'MAE': mae,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"{name} - R² Score: {r2:.3f}")
    print(f"{name} - Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"{name} - MAE: {mae:.3f} hours")

# 8. Model Comparison
print("\n=== Model Comparison Results ===")
print(f"{'Model':<20} {'R² Score':<10} {'MAE (hours)':<12} {'CV R²':<15}")
print("-" * 60)

for name, results in model_results.items():
    print(f"{name:<20} {results['R2']:<10.3f} {results['MAE']:<12.3f} {results['CV_R2_mean']:.3f}±{results['CV_R2_std']:.3f}")

# Find best model
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['R2'])
best_model = trained_models[best_model_name]
print(f"\nBest performing model: {best_model_name}")

# 9. Feature Importance Analysis (for Linear Regression)
print(f"\n=== Feature Importance Analysis ===")
if hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Coefficient': best_model.coef_
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10)[['Feature', 'Coefficient']])
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(10)
    colors = ['red' if x < 0 else 'blue' for x in top_features['Coefficient']]
    plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved as 'feature_importance.png'")

# 10. Model Performance Visualization
print("\n=== Creating Performance Visualizations ===")

# Model comparison bar plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# R² comparison
models_list = list(model_results.keys())
r2_scores = [model_results[model]['R2'] for model in models_list]
bars1 = ax1.bar(models_list, r2_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance Comparison - R² Score')
ax1.set_ylim(0, 1)
for i, bar in enumerate(bars1):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{r2_scores[i]:.3f}', ha='center', va='bottom')

# MAE comparison
mae_scores = [model_results[model]['MAE'] for model in models_list]
bars2 = ax2.bar(models_list, mae_scores, color=['#3498db', '#e74c3c', '#2ecc71'])
ax2.set_ylabel('Mean Absolute Error (hours)')
ax2.set_title('Model Performance Comparison - MAE')
for i, bar in enumerate(bars2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{mae_scores[i]:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Predictions vs Actual plot for best model
plt.figure(figsize=(10, 8))
best_predictions = model_results[best_model_name]['predictions']
plt.scatter(y_test, best_predictions, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Study Hours')
plt.ylabel('Predicted Study Hours')
plt.title(f'{best_model_name}: Predicted vs Actual Study Hours')
plt.grid(alpha=0.3)

# Add R² score to plot
r2_text = f'R² = {model_results[best_model_name]["R2"]:.3f}'
plt.text(0.05, 0.95, r2_text, transform=plt.gca().transAxes, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

print("Performance visualizations saved: model_comparison.png, predictions_vs_actual.png")

# 11. Practical Examples
print("\n=== Practical Examples ===")
print("Let's predict study hours for some example students:")

# Create example students
example_students = pd.DataFrame({
    'G1': [15, 8, 12],
    'G2': [16, 9, 13],
    'failures': [0, 1, 0],
    'absences': [2, 8, 4],
    'age': [17, 18, 16],
    'Medu': [4, 2, 3],
    'Fedu': [4, 2, 3],
    'studytime': [3, 2, 2],
    'schoolsup': [0, 1, 0],  # 0=no, 1=yes (already encoded)
    'famsup': [1, 0, 1],
    'paid': [0, 1, 0],
    'activities': [1, 0, 1],
    'nursery': [1, 1, 0],
    'higher': [1, 1, 1],
    'internet': [1, 0, 1],
    'romantic': [0, 1, 0],
    'freetime': [3, 4, 2],
    'goout': [2, 4, 3],
    'Dalc': [1, 3, 1],
    'Walc': [2, 4, 2],
    'health': [4, 3, 5]
})

# Make sure we have all features
for feature in available_features:
    if feature not in example_students.columns:
        example_students[feature] = 0  # Default value

# Reorder columns to match training data
example_students = example_students[available_features]

# Scale the examples
example_scaled = scaler.transform(example_students)

# Make predictions
example_predictions = best_model.predict(example_scaled)

student_profiles = [
    "High performer (G1=15, G2=16, no failures, low absences)",
    "Struggling student (G1=8, G2=9, 1 failure, high absences)",
    "Average student (G1=12, G2=13, no failures, moderate absences)"
]

for i, (profile, hours) in enumerate(zip(student_profiles, example_predictions)):
    print(f"Student {i+1}: {profile}")
    print(f"Predicted study hours needed: {hours:.1f} hours")
    print()

# 12. Summary
print("=== Project Summary ===")
print(f"✓ Successfully reverse-engineered UCI Student Performance Dataset")
print(f"✓ Created target variable: study hours needed (range: {y.min():.1f}-{y.max():.1f} hours)")
print(f"✓ Trained and compared 3 regression models")
print(f"✓ Best model: {best_model_name} (R² = {model_results[best_model_name]['R2']:.3f})")
print(f"✓ Generated 5 visualization files")
print(f"✓ Model can predict study hours within ±{model_results[best_model_name]['MAE']:.1f} hours on average")

print("\nGenerated files:")
files = ['correlation_matrix.png', 'study_hours_distribution.png', 'grades_vs_study_hours.png', 
         'feature_importance.png', 'model_comparison.png', 'predictions_vs_actual.png']
for file in files:
    print(f"  - {file}")

print(f"\nProject completed successfully! 🎓")