import joblib
import os
import numpy as np
 
# Load the pre-trained classifier from a pickle file
LOADED_CLASSIFIER = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model/caTCRem.pkl'))
 
def predict_task(data):
    """
    Predict the probabilities for each class for a single test sample.
    
    Parameters:
    data (list or array-like): The test sample data (after encoding).
    
    Returns:
    float: The maximum probability score.
    """
    # Obtain the probabilities for each class for the given test sample
    probabilities = LOADED_CLASSIFIER.predict_proba([data])[0]
 
    # Find the index and value of the maximum probability
    max_score_index = np.argmax(probabilities)  # Get the index of the maximum probability
    max_score = probabilities[max_score_index]  # Get the value of the maximum probability
 
    return max_score
 
# Define the file path to read the normalized scores
file_path = '../result/res.txt'
normalized_scores_list = []
 
# Read the normalized scores from the file
with open(file_path, 'r') as file:
    for line in file:
        normalized_scores_list.append(float(line.strip()))
 
score = predict_task(normalized_scores_list)  # Predict
 
# Print the probability of cancer (note: this will just print the score)
print('The probability of cancer is: ' + str(score))