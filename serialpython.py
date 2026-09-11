# pip install pyserial
import serial

ser = serial.Serial('COM3', 9600)

# Parse one line of data like: "Temp:26.30,Gas:180.00,Awake:1,Fan:1,Light:0"
# string manipulation
def parse_line(line):
    data = {}
    pairs = line.split(',')
    for pair in pairs:
        if ':' in pair:
            key, value = pair.split(':')
            data[key.strip()] = value.strip() # Removes extra spaces
    return data # The values are strings

def cry(input):
    input= input+ '\r'
    ser.write(input.encode())
