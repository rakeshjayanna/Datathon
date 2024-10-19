import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('/content/data.csv')  # Replace 'data.csv' with the path to your dataset
print(df.head)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Dropping columns that are not useful for prediction
df = df.drop(columns=['id', 'Unnamed: 32'])

# Convert diagnosis values: M (Malignant) = 1, B (Benign) = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

sns.lmplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df)
plt.title('Radius vs Texture by Diagnosis')
plt.show()

sns.lmplot(x='smoothness_mean', y='compactness_mean', hue='diagnosis', data=df)
plt.title('Smoothness vs Compactness by Diagnosis')
plt.show()

# Define input (features) and output (target) data
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors

# Fit the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(y_test, predictions))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)

# Perform cross-validation
cv_scores = cross_val_score(knn, X, y, cv=10)  # 10-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {cv_scores.mean():.2f}')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Assuming you have already loaded your dataset and preprocessed it
# Define input (features) and output (target) data
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate misclassification error for different values of k
error = []
k_values = range(1, 21)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    pred_k = knn.predict(X_test)
    error.append(np.mean(pred_k != y_test))  # Calculate misclassification error

# Plotting the misclassification error
plt.figure(figsize=(10, 6))
plt.plot(k_values, error, marker='o')
plt.title('Misclassification Error vs. k')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.xticks(k_values)
plt.grid()
plt.show()
