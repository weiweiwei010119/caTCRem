# Load the data from a tab-separated values file
ref_data <- read.csv('.../data/accurateTCR_sort.tsv', sep = "\t")

# 1. Remove non-human data
human_index <- ref_data$Species == 'HomoSapiens'
ref_data <- ref_data[human_index, ]

# 2. Identify and remove duplicate CDR3 sequences, keeping only the first occurrence
repeat_indices <- list()  # Initialize an empty list to store indices of duplicate entries (second occurrence onwards)
arr <- ref_data$CDR3
 
# Iterate through the array
for (i in seq_along(arr)) {
  # Get the current element's value
  current_value <- arr[i]
  
  # Check if the current element has appeared before (i.e., is a duplicate)
  prev_indices <- seq_along(arr)[arr == current_value & seq_along(arr) < i]
  
  # If it has appeared before and is not the first occurrence (i.e., length > 0)
  if (length(prev_indices) > 0) {
    # Add the current index to the list of duplicate indices
    repeat_indices <- c(repeat_indices, i)
  }
}
 
# Unlist the indices of duplicate entries
dup_index <- unlist(repeat_indices)
 
# Extract the unique CDR3 sequences (excluding duplicates)
ref <- ref_data$CDR3[-dup_index]
 
# Write the unique CDR3 sequences to a CSV file
write.csv(ref, '.../data/CDR3.csv', quote = FALSE)
