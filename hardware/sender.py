import serial
import time

PORT = "COM12"       # Change this
BAUD = 115200

# Read moves from file
with open("moves.txt", "r") as f:
    moves = [line.strip() for line in f if line.strip()]

# Send over serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)  # Wait for ESP to reset

for move in moves:
    print(f"Sending: {move}")
    ser.write((move + "\n").encode())
    time.sleep(1)

ser.close()
