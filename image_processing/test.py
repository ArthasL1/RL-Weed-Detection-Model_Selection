import numpy as np

def process_optimal_reward(L0, L1, L2, L3):
    # Convert the lists to numpy arrays for easier manipulation
    L0 = np.array(L0)
    L1 = np.array(L1)
    L2 = np.array(L2)
    L3 = np.array(L3)

    # Stack the lists along a new dimension to facilitate comparison
    stacked = np.stack([L0, L1, L2, L3], axis=0)

    # Find the maximum values across the new dimension (axis=0) 
    max_values = np.max(stacked, axis=0)

    # Find the indices of the maximum values across the new dimension (axis=0)
    max_indices = np.argmax(stacked, axis=0)

    # Calculate the sum of each sublist (row-wise sum) to reduce the 2D list to 1D
    row_sums = np.sum(max_values, axis=1)

    # Calculate the average of the summed values
    average_value = np.mean(row_sums)

    return max_values.tolist(), max_indices.tolist(), average_value



# Example usage:
L0 = [
    [1, 5, 3],
    [7, 2, 6]
]

L1 = [
    [4, 2, 9],
    [1, 3, 5]
]

L2 = [
    [0, 8, 1],
    [3, 4, 7]
]

L3 = [
    [6, 3, 2],
    [8, 9, 0]
]

# Call the function
max_list, indices_list, average = process_optimal_reward(L0, L1, L2, L3)

# Output the results
print("New list with maximum values:", max_list)
print("Indices list:", indices_list)
print("Average of summed rows:", average)