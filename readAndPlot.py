import serial
import numpy as np
import matplotlib.pyplot as plt

# Function to read 8 rows of data via serial communication after receiving "Print"
def read_matrix_from_serial(serial_port):
    ser = serial.Serial(serial_port, baudrate=115200, timeout=1)
    
    # Set up the plot for real-time updates
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    img = ax.imshow(np.zeros((8, 8)), cmap='RdYlGn', vmin=0, vmax=1000)  # Initialize with an empty matrix

    while True:
        # Wait for the "Print" message to start reading a new matrix
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(f"{line}")
            
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
                            rows.append(row)
                
                # Convert the list of rows into a NumPy array (our 8x8 matrix)
                matrix = np.array(rows)
                
                # Update the image data with the new matrix
                img.set_data(matrix)
                
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
