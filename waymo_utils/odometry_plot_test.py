import matplotlib.pyplot as plt
import numpy as np

# Sample vehicle pose data for demonstration
vehicle_pose_data = [
    [3827.279206947279, 3436.145519070632, -62.32],
    [3827.279129648357, 3436.1456194899533, -62.32],
    [3827.2791627197203, 3436.1454609015495, -62.32],
    [3827.279228771735, 3436.145283828503, -62.32],
    [3827.2793138487536, 3436.1453435221706, -62.32]
]

# Extracting positions
positions = np.array(vehicle_pose_data)

# Creating 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the vehicle positions
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', marker='o')

# Plotting the path
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c='r')

# Setting labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Setting title
ax.set_title('Vehicle Pose Path')

# Show the plot
plt.show()
