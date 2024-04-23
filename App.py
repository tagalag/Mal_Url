import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset locally
data = pd.read_csv('data.csv')

# Encode the label ('good', 'bad') as integers
data['label'] = data['label'].map({'good': 1, 'bad': 0})

# Extract features from URLs to create a feature matrix
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['url'])

# Target variable
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=1)

# Note: Scaling is generally not necessary for CountVectorizer output and can be omitted here

# Train the KNN model
k = 7
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict the test set results
y_pred = knn.predict(X_test)

# Print the confusion matrix and accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print('KNN with k =', k)
print('Confusion Matrix:')
print(cm)
print('Accuracy:', accuracy)