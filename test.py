import numpy as np

def center_of_mass(arr):
    rows, cols = arr.shape
    total_mass = arr.sum()
    center_of_mass_x = np.sum(np.arange(cols) * arr) / total_mass
    center_of_mass_y = np.sum(np.arange(rows) * arr.T) / total_mass
    return center_of_mass_x, center_of_mass_y

def shift_center_of_mass(arr):
    center_x, center_y = center_of_mass(arr)
    rows, cols = arr.shape
    target_center_x = (cols - 1) / 2
    target_center_y = (rows - 1) / 2
    shift_x = int(target_center_x - center_x)
    shift_y = int(target_center_y - center_y)

    shifted_arr = np.roll(arr, shift_x, axis=1)
    shifted_arr = np.roll(shifted_arr, shift_y, axis=0)

    return shifted_arr

# Example usage:
if __name__ == "__main__":
    # Create a sample 2D array (replace this with your own data)
    sample_array = np.array([[0, 0, 0, 0, 0],
                             [1, 1, 1, 0, 0],
                             [1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 0]])

    print("Original Array:")
    print(sample_array)

    shifted_array = shift_center_of_mass(sample_array)
    print("\nShifted Array with Center of Mass at the Center:")
    print(shifted_array)
