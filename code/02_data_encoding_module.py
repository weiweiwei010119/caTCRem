import pandas as pd
from textdistance import levenshtein  # Import the levenshtein distance function from the textdistance library
 
# Specify the path to the CSV file containing the data you need to process
file_path = 'data.csv'
 
# Specify the path to the reference dataset CSV file containing accurate CDR3 sequences
accurate_cdr3_path = '../data/CDR3.csv'
# Read the accurate CDR3 sequences into a DataFrame
cdr3_list = pd.read_csv(accurate_cdr3_path)
 
try:
    # Try to read the data from the specified CSV file
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: The specified file was not found.")
else:
    # Extract the CDR3 sequences and their frequencies from the DataFrame
    cdr3_to_compare = df['cdr3']
    cdr3_freq_list = df['freq']
    
    # Check if CDR3 sequences are provided
    if not cdr3_to_compare.any():
        print('No CDR3 sequences provided.')
    
    # Check if CDR3 frequencies are provided
    if not cdr3_freq_list.any():
        print('No CDR3 frequencies provided.')
 
    # Convert the frequency list to a list of floats, setting non-convertible values to 0
    cdr3_freq = []
    for num in cdr3_freq_list:
        try:
            cdr3_freq.append(float(num))
        except ValueError:
            cdr3_freq.append(0)
 
    # Check if the lengths of the CDR3 sequence list and frequency list match
    if len(cdr3_to_compare) != len(cdr3_freq):
        print('The lengths of CDR3 sequences and frequencies do not match.')
 
    # Calculate the score for each accurate CDR3 sequence
    scores = []
    for accurate_cdr3 in cdr3_list['CDR3_column_name']:  # Assuming there's a specific column name for CDR3 in cdr3_list
        score = 0
        for i in range(len(cdr3_to_compare)):
            cdr3_to_compare_cdr3 = cdr3_to_compare[i]
            cdr3_freq_cdr3 = cdr3_freq[i]
            lv = levenshtein(accurate_cdr3, cdr3_to_compare_cdr3)  # Calculate the Levenshtein distance
            if lv <= 1:  # If the distance is 1 or less, add the frequency to the score
                score += cdr3_freq_cdr3
        scores.append(score)
 
    # Normalize the scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        normalized_scores = [0.0] * len(scores)  # If all scores are the same, set them to 0 after normalization
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]  # Normalize scores to the range [0, 1]
 
    # Save the normalized scores to a text file
    txt_file_path = '../result/res.txt'
    with open(txt_file_path, 'w') as file:
        for score in normalized_scores:
            file.write(f"{score}\n")  # Write each normalized score to a new line in the text file