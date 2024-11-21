import serial
import time

# Replace '/dev/tty.usbserial-XXXX' with your USB port path
# You can find the path using `ls /dev/tty.*` in your terminal
serial_port = '/dev/tty.usbmodem1101'
baud_rate = 115200  # Adjust based on your device's requirements

try:
    # Open the serial port
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    print(f"Connected to {serial_port} at {baud_rate} baud.")

    # Continuously read data
    while True:
        if ser.in_waiting > 0:
            # Read a line of data from the serial port
            data = ser.readline().decode('utf-8').strip()
            print(f"Received: {data}")

        # Sleep to avoid overloading the CPU
        time.sleep(0.1)

except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nExiting program.")
finally:
    # Close the serial port if open
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
