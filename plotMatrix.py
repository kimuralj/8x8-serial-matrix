import numpy as np
import matplotlib.pyplot as plt

# Simulating an 8x8 matrix with random values for demonstration
matrix = np.random.randint(0, 1200, (8, 8))

# Define a function to plot the matrix with a gradient colormap
def plot_matrix_with_gradient(matrix):
    # Set up the plot
    plt.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1000, interpolation='nearest')
    
    # Add color bar to show the value scale
    plt.colorbar(label='Value')
    
    # Turn off axis labels for a cleaner look
    plt.axis('off')
    
    # Add title
    plt.title('8x8 Matrix with Gradient Color')
    
    # Show the plot
    plt.show()

# Call the plotting function
plot_matrix_with_gradient(matrix)
