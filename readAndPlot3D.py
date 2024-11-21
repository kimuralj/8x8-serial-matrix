import serial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom

# Function to read 8 rows of data via serial communication after receiving "Print"
def read_matrix_from_serial(serial_port):
    ser = serial.Serial(serial_port, baudrate=115200, timeout=1)
    
    # Set up the plot for real-time updates
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create x and y coordinates for the grid (row and column indices)
    x = np.arange(0, 32)
    y = np.arange(0, 32)
    x, y = np.meshgrid(x, y)  # Create a meshgrid for plotting
    
    # Initialize a surface plot with an empty matrix
    matrix = np.zeros((32, 32))
    surf = ax.plot_surface(x, y, matrix, cmap='RdYlGn', edgecolor='none')
    
    # Add color bar to show the value scale
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('X Axis (Columns)')
    ax.set_ylabel('Y Axis (Rows)')
    ax.set_zlabel('Z Axis (Values)')
    
    # Set fixed Z-axis limits (e.g., from 0 to 1000)
    ax.set_zlim(0, 300)

    isFirst = True

    while True:
        # Wait for the "Print" message to start reading a new matrix
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            # print(f"{line}")
            print()
            
            if line[0] == "P":
                print("Starting to read new matrix...")
                rows = []  # Initialize an empty list to store rows
                
                # Read 8 rows for the matrix
                while len(rows) < 8:
                    if ser.in_waiting > 0:
                        row_line = ser.readline().decode('utf-8').strip()
                        row = list(map(int, row_line.split(',')))  # Convert row to a list of integers
                        print(row)
                        
                        if len(row) == 8:  # Ensure it's a valid row
                            rows.append(np.array(row))
                
                # Convert the list of rows into a NumPy array (our 8x8 matrix)
                matrix = np.array(rows)

                matrix = zoom(matrix, zoom=4, order=1)

                if isFirst:
                    firstMatrix = matrix
                    isFirst = False

                else:
                    matrix = abs(matrix - firstMatrix)

                    # Limit max value
                    matrix = np.clip(matrix, a_min=0, a_max=300)
                    
                    # Clear the previous surface plot
                    ax.clear()
                    
                    # Plot the new surface with updated data
                    surf = ax.plot_surface(x, y, matrix, cmap='RdYlGn', edgecolor='none')

                    # Set fixed Z-axis limits (e.g., from 0 to 1000)
                    ax.set_zlim(0, 300)
                    
                    # Redraw the canvas to update the plot in real-time
                    fig.canvas.draw()
                    fig.canvas.flush_events()

# Main function to continuously read and plot matrices
def main():
    # Replace serial_port with your actual serial port (e.g., '/dev/ttyUSB0' on Linux or 'COM3' on Windows)
    serial_port = '/dev/tty.usbmodem1101'
    
    # Continuously read and plot new matrices from serial communication
    read_matrix_from_serial(serial_port)

if __name__ == "__main__":
    main()
