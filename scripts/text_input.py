from pythonosc.udp_client import SimpleUDPClient
import time

# specify the IP address and port number for the OSC server
ip = "127.0.0.1"
port = 4976

# create a client instance
client = SimpleUDPClient(ip, port)

# infinite loop to get input and send it as an OSC message
while True:
    try:
        # get input from the terminal
        input_text = input("Enter the text to be sent over OSC: ")

        # send the message "/message" with the input as argument
        client.send_message("/message", input_text)

        # sleep for a bit to manage the pace of sending messages
        time.sleep(0.01)

    except KeyboardInterrupt:
        print("Program interrupted by user, exiting.")
        break
