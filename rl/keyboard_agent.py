import sys
import os
import pygame
import gymnasium as gym
import warnings
import numpy as np
import torch
import rclpy


warnings.filterwarnings("ignore")

# Initialize pygame and set up a window for receiving input
# pygame.init()
# screen = pygame.display.set_mode((400, 300))  # Pygame window (400x300)
# pygame.display.set_caption("Control Lunar Lander")

# Set up task and environment
# task = task_config()
# task.headless = False
num_envs = 4
env = gym.make("gymnasium_arg:blueboat-v1", num_envs=num_envs, world="waves", veh='blueboat')
env.close()
while True:
    pass
# env.reset()
# env.render()

# Define key mapping for actions
# def get_keyboard_action():
#     keys = pygame.key.get_pressed()

#     # Default action: no movement
#     action = torch.zeros(num_envs, 4, device=task.device)

#     # Map keys to actions
#     if keys[pygame.K_UP]:
#         action[0][0] = 1.0  # Thrust up
#     if keys[pygame.K_LEFT]:
#         action[0][1] = 1.0  # Thrust left
#     if keys[pygame.K_RIGHT]:
#         action[0][2] = 1.0  # Thrust right
#     if keys[pygame.K_DOWN]:
#         action[0][3] = 1.0  # Thrust down (if applicable)

#     return action

# # Main loop
# clock = pygame.time.Clock()
# while True:
#     # Handle pygame events
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()

#     # Fill the screen with a background color (optional)
#     screen.fill((30, 30, 30))  # Dark gray background for the pygame window

#     # Get keyboard input action
#     action = get_keyboard_action()

#     # Step the environment
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("Reward: ", reward)

#     # Render the environment (this will render in a separate window, depending on your environment's setup)
#     env.render()

#     # Reset the environment if needed
#     for i in range(len(terminated)):
#         if terminated[i] or truncated[i]:
#             env.reset_idx(i)

#     # Update the pygame display
#     pygame.display.flip()

#     # Control frame rate (optional, to make keyboard control more manageable)
#     clock.tick(30)  # Limit to 30 frames per second