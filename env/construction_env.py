import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import sys

class ConstructionEnv(gym.Env):
    """
    2D PyGame Environment for Multi-Agent Construction Scenario.
    """
    def __init__(self, render=False, num_robots=4):
        super(ConstructionEnv, self).__init__()
        self.render_mode = render
        self.num_robots = num_robots
        
        # Grid/World settings
        self.main_width, self.main_height = 800, 600
        self.hud_width = 250
        self.width = self.main_width + self.hud_width
        self.height = self.main_height
        
        # Action space: 0: Idle, 1: North, 2: South, 3: East, 4: West
        self.action_space = spaces.Dict({
            f"robot_{i}": spaces.Discrete(5) for i in range(num_robots)
        })
        
        # Observation space: [x, y, vx, vy, battery, has_material]
        self.observation_space = spaces.Dict({
            f"robot_{i}": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) 
            for i in range(num_robots)
        })

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("MARL Construction Site - Premium Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 18)
            self.bold_font = pygame.font.SysFont("Arial", 20, bold=True)
            
            # Load Assets
            try:
                self.ground_tex = pygame.image.load("assets/ground.png")
                self.ground_tex = pygame.transform.scale(self.ground_tex, (self.main_width, self.main_height))
                self.robot_tex = pygame.image.load("assets/robot.png")
                self.robot_tex = pygame.transform.scale(self.robot_tex, (40, 40))
            except:
                self.ground_tex = None
                self.robot_tex = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Initialize Robot States
        self.robot_states = []
        for i in range(self.num_robots):
            self.robot_states.append({
                "pos": np.array([np.random.uniform(100, self.main_width-100), 
                                 np.random.uniform(100, self.main_height-100)], dtype=np.float32),
                "battery": 100.0,
                "has_material": 0.0,
                "distance": 0.0
            })
            
        # Static Areas (Circles/Rects)
        self.storage_pos = np.array([100, 100])
        self.construction_zones = [
            {"pos": np.array([700, 100]), "label": "Zone A", "progress": 0},
            {"pos": np.array([700, 300]), "label": "Zone B", "progress": 0},
            {"pos": np.array([700, 500]), "label": "Zone C", "progress": 0}
        ]
        
        return self._get_obs()

    def step(self, actions):
        speed = 4.0
        for i in range(self.num_robots):
            action = actions.get(f"robot_{i}", 0)
            old_pos = self.robot_states[i]["pos"].copy()
            
            # Movement
            if action == 1: # North
                self.robot_states[i]["pos"][1] -= speed
            elif action == 2: # South
                self.robot_states[i]["pos"][1] += speed
            elif action == 3: # East
                self.robot_states[i]["pos"][0] += speed
            elif action == 4: # West
                self.robot_states[i]["pos"][0] -= speed
            
            # Boundary checks (within main site)
            self.robot_states[i]["pos"] = np.clip(self.robot_states[i]["pos"], [20, 20], [self.main_width-20, self.main_height-20])
            
            # Distance tracking
            self.robot_states[i]["distance"] += np.linalg.norm(self.robot_states[i]["pos"] - old_pos) / 10.0
            
            # Interaction with Storage
            if np.linalg.norm(self.robot_states[i]["pos"] - self.storage_pos) < 40:
                if self.robot_states[i]["has_material"] < 1.0:
                    self.robot_states[i]["has_material"] = 1.0
                    self.robot_states[i]["battery"] = min(100, self.robot_states[i]["battery"] + 10) # Recharge slightly at storage
            
            # Interaction with Zones
            for zone in self.construction_zones:
                if np.linalg.norm(self.robot_states[i]["pos"] - zone["pos"]) < 40:
                    if self.robot_states[i]["has_material"] > 0:
                        self.robot_states[i]["has_material"] = 0
                        zone["progress"] += 1
            
            # Battery drain
            if action != 0:
                self.robot_states[i]["battery"] -= 0.08
            else:
                self.robot_states[i]["battery"] -= 0.01 # Idle drain
        
        if self.render_mode:
            self.render()
            
        obs = self._get_obs()
        rewards = {f"robot_{i}": 0.0 for i in range(self.num_robots)}
        terminations = {f"robot_{i}": self.robot_states[i]["battery"] <= 0 for i in range(self.num_robots)}
        truncations = {f"robot_{i}": False for i in range(self.num_robots)}
        
        return obs, rewards, terminations, truncations, {}

    def _get_obs(self):
        obs = {}
        for i in range(self.num_robots):
            s = self.robot_states[i]
            obs[f"robot_{i}"] = np.array([
                s["pos"][0] / self.main_width,
                s["pos"][1] / self.main_height,
                0.0, 0.0,
                s["battery"] / 100.0,
                s["has_material"]
            ], dtype=np.float32)
        return obs

    def render(self):
        # 1. Main Construction Area
        if self.ground_tex:
            self.screen.blit(self.ground_tex, (0, 0))
        else:
            self.screen.fill((180, 160, 140), rect=(0, 0, self.main_width, self.main_height)) # Dirt color
        
        # Draw some "grid lines" or dust
        for x in range(0, self.main_width, 100):
            pygame.draw.line(self.screen, (160, 140, 120), (x, 0), (x, self.main_height), 1)
        for y in range(0, self.main_height, 100):
            pygame.draw.line(self.screen, (160, 140, 120), (0, y), (self.main_width, y), 1)

        # Draw Storage (Industrial blue)
        pygame.draw.circle(self.screen, (30, 80, 150), self.storage_pos, 45, 3)
        pygame.draw.circle(self.screen, (50, 100, 200), self.storage_pos, 35)
        lbl = self.font.render("MATERIAL STORAGE", True, (255, 255, 255))
        self.screen.blit(lbl, (self.storage_pos[0]-70, self.storage_pos[1]+45))
        
        # Draw construction zones (Construction orange)
        for zone in self.construction_zones:
            # Outer border
            z_rect = (zone["pos"][0]-35, zone["pos"][1]-35, 70, 70)
            pygame.draw.rect(self.screen, (255, 140, 0), z_rect, 4)
            # Inner fill based on progress
            prog_height = int(70 * min(1.0, zone["progress"]/50.0))
            if prog_height > 0:
                pygame.draw.rect(self.screen, (200, 100, 0), (zone["pos"][0]-35, zone["pos"][1]+35-prog_height, 70, prog_height))
            
            lbl = self.font.render(zone["label"], True, (50, 50, 50))
            self.screen.blit(lbl, (zone["pos"][0]-30, zone["pos"][1]-55))

        # Draw robots
        for i in range(self.num_robots):
            pos = self.robot_states[i]["pos"]
            if self.robot_tex:
                # Rotate robot conceptually? For now just blit
                self.screen.blit(self.robot_tex, (int(pos[0]-20), int(pos[1]-20)))
            else:
                color = (255, 0, 0) if i < self.num_robots//2 else (0, 200, 0)
                pygame.draw.circle(self.screen, color, pos.astype(int), 18)
            
            # Carry indicator
            if self.robot_states[i]["has_material"] > 0:
                pygame.draw.rect(self.screen, (100, 50, 20), (pos[0]-5, pos[1]-5, 10, 10))

        # 2. HUD Area
        hud_rect = (self.main_width, 0, self.hud_width, self.main_height)
        pygame.draw.rect(self.screen, (40, 44, 52), hud_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (self.main_width, 0), (self.main_width, self.main_height), 2)
        
        title = self.bold_font.render("CONSTRUCTION FLEET", True, (255, 200, 0))
        self.screen.blit(title, (self.main_width + 20, 20))
        
        for i in range(self.num_robots):
            y_off = 70 + i * 85
            s = self.robot_states[i]
            
            # Robot ID
            r_id = self.font.render(f"Robot {i}: Status", True, (200, 200, 200))
            self.screen.blit(r_id, (self.main_width + 20, y_off))
            
            # Battery Bar
            bat_color = (0, 255, 0) if s["battery"] > 30 else (255, 0, 0)
            pygame.draw.rect(self.screen, (60, 60, 60), (self.main_width + 20, y_off + 25, 200, 12))
            pygame.draw.rect(self.screen, bat_color, (self.main_width + 20, y_off + 25, int(s["battery"] * 2), 12))
            
            # Stats Text
            stats = self.font.render(f"Load: {'[X]' if s['has_material'] > 0 else '[ ]'} Dist: {s['distance']:.1f}m", True, (150, 150, 150))
            self.screen.blit(stats, (self.main_width + 20, y_off + 45))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == "__main__":
    # Test execution
    print("Starting Premium Visualization... Close window to exit.")
    env = ConstructionEnv(render=True, num_robots=6)
    env.reset()
    running = True
    while running:
        # User input / Close check
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Simple random-walk logic for visualization
        actions = {f"robot_{i}": np.random.randint(0, 5) for i in range(env.num_robots)}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Reset any robot that dies
        if any(terminations.values()):
            env.reset()
            
    env.close()
