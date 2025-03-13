import matplotlib.pyplot as plt
import numpy as np
import socket
import json
from mpl_toolkits.mplot3d import Axes3D

# Enable interactive mode
plt.ion()

# Create a static 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
sc = ax.scatter([], [], [], c='r', marker='o')

plt.draw()
plt.pause(0.1)  # Allow window to update

# Socket setup
HOST = '127.0.0.1'  
PORT = 65432        
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST, PORT))
sock.listen(1)
print("3D Visualization Started. Waiting for data...")

conn, addr = sock.accept()
print("🎉 Connection Established!")

# Live updating loop
while True:
    try:
        data = conn.recv(1024).decode()
        if not data:
            continue

        coords = json.loads(data)
        x, y, z = coords["x"], coords["y"], coords["z"]

        # Clear previous points and replot
        sc._offsets3d = (np.array([x]), np.array([y]), np.array([z]))
        plt.draw()
        plt.pause(0.05)  # Give time for UI updates

    except json.JSONDecodeError:
        print("❌ Error: Received invalid JSON data")
    except KeyboardInterrupt:
        print("Exiting...")
        break

conn.close()
sock.close()
plt.ioff()  # Turn off interactive mode
plt.show()
