import pandas as pd
from sklearn import preprocessing 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier
import random
from tqdm import tqdm
import sklearn

# Load the dataset, setting the first column as the index
data = pd.read_csv('../data/dataset.csv', index_col=0)
# Drop rows with missing values
data = data.dropna()

# Encode the 'kind' column using LabelEncoder
le = preprocessing.LabelEncoder()
le.fit(data['kind'].values)  # Fit the encoder to the 'kind' column values
print(le.classes_)  # Print the classes found in the 'kind' column
 
# Transform the 'kind' column to encoded values, then invert the encoding (1 - value)
data['kind'] = 1 - le.transform(data['kind'].values)
 
# Split the data into training and testing sets based on the 'model' column
train_data = data[data['model'] == 'train']
test_data = data[data['model'] != 'train']
 
# Define features (X) and target (y) for training and testing data
x_train = train_data.iloc[:, :-2].values  
y_train = train_data.iloc[:, 2625].values 
x_test = test_data.iloc[:, :-2].values  
y_test = test_data.iloc[:, 2625].values  
 
# Initialize lists to store accuracy and AUC scores
acc = []
auc = []
n = 100  # Number of features (assuming we want to randomly select features for each iteration)
 
# Loop 100 times to perform random feature selection and model evaluation
for i in tqdm(range(100)):
    # Randomly select 'n' features (in this case, it selects all features but in a random order)
    ran = [random.randint(0, x_train.shape[1] - 1) for _ in range(n)]
    x_train_selected = x_train[:, ran]  # Select random features for training data
    x_test_selected = x_test[:, ran]  # Select corresponding random features for testing data
    
    # Initialize and train the RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=199, random_state=13, max_depth=4)
    classifier.fit(x_train_selected, y_train)
    
    # Predict the labels for the test set
    y_pred = classifier.predict(x_test_selected)
    
    # Calculate and append accuracy and AUC scores (note: AUC is not typically used for classification without probabilities)
    acc.append(sklearn.metrics.accuracy_score(y_test, y_pred))
    
# Create a DataFrame to store the results
df = pd.DataFrame({'acc': acc})  # Remove 'auc' unless using probabilities and a suitable metric
df.to_csv('../res/' + str(n) + '_fea.csv', index=False)  # Adjust the path and filename as needed