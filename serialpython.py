# pip install pyserial
import serial

port = 'COM3'
baudrate = 9600

ser = serial.Serial(port, baudrate)

# Parse one line of data like: "Temp:26.30,Gas:180.00,Awake:1,Fan:1,Light:0"
def parse_line(line):
    data = {}
    pairs = line.split(',')
    for pair in pairs:
        if ':' in pair:
            key, value = pair.split(':')
            data[key.strip()] = value.strip()
    return data
