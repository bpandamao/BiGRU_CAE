import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create a random matrix of shape (200, 1000)
matrix = np.load("./sig_toy_signal.npy")[:, 50000:]

# Set up the figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

# Set the limits of the plot
ax.set_xlim(0, matrix.shape[1] - 1)  # X limits from 0 to 999
ax.set_ylim(np.min(matrix), np.max(matrix))  # Y limits based on matrix values
ax.set_xlabel('Column Index')
ax.set_ylabel('Value')
ax.set_title('Row Animation of a Matrix')

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Animation function
def animate(i):
    line.set_data(range(matrix.shape[1]), matrix[i])
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, animate, frames=matrix.shape[0], init_func=init, blit=True, repeat=False)

# Show the animation
plt.show()