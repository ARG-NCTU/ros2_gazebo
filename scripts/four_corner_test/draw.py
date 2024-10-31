import matplotlib.pyplot as plt
import numpy as np
import math

# Load data from file
positions = []
headings = []
square_corners = np.array([[0, 0, 0], [10, 0, -0.5], [10, 10, 0.5], [0, 10, 1.0], [0, 0, -1.0]])

with open("./blueboat_four.txt") as file:
    for line in file:
        if "position:" in line:
            x = float(next(file).split(":")[1])
            y = float(next(file).split(":")[1])
            positions.append((x, y))
        elif "orientation:" in line:
            qx = float(next(file).split(":")[1])
            qy = float(next(file).split(":")[1])
            qz = float(next(file).split(":")[1])
            qw = float(next(file).split(":")[1])
            heading = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
            headings.append(heading)

# Plot the trajectory
fig, ax = plt.subplots()
for i in range(1, len(positions), 10):
    x1, y1 = positions[i-1]
    x2, y2 = positions[i]
    
    # Plot points with markers
    ax.plot([x1, x2], [y1, y2], color="pink", marker="o", markersize=2)

    # Add heading arrow in a different color
    dx = 0.2 * math.cos(headings[i])
    dy = 0.2 * math.sin(headings[i])
    ax.arrow(x2, y2, dx, dy, head_width=0.05, head_length=0.1, color="red")

# Plot the square and add 10 green arrows along each side
ax.plot(square_corners[:, 0], square_corners[:, 1], color="green", linewidth=2)
for i in range(len(square_corners) - 1):
    start = square_corners[i]
    end = square_corners[i + 1]
    x_start, y_start, _ = start
    x_end, y_end, angle = end
    
    # Calculate interval points between the corners
    for j in range(10):  # Generate 10 points along each side
        x = x_start + (x_end - x_start) * j / 9  # 9 intervals for 10 points
        y = y_start + (y_end - y_start) * j / 9
        dx = 0.2 * math.cos(angle)
        dy = 0.2 * math.sin(angle)
        ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, color="green")

# Plot settings
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("2D Trajectory with Heading Arrows and Square Corner Arrows")

plt.show()
