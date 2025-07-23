import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random
import re
import numba

# Stable Baselines3 for Reinforcement Learning
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

# ==============================================================================
# 1. HYPERPARAMETERS AND CONFIGURATION (V2 - High Ground AI)
# ==============================================================================

# --- Training Hyperparameters ---
TOTAL_TIMESTEPS = 20_000_000 # Increased for more complex task
NUM_ENVIRONMENTS = 16
LEARNING_RATE = 0.0001 # Lower learning rate for more stable learning with CNN
N_STEPS = 4096
BATCH_SIZE = 512 # Smaller batch size is often better for CNNs

# --- Simulation Configuration ---
TROOP_COUNT = 500
NUM_OBSTACLES = 10
MAX_STEPS_PER_EPISODE = 3000

# --- V2 Observation Grid ---
GRID_SIZE = 16 # We'll use a 16x16 grid

# --- File Paths ---
MODEL_SAVE_PATH = "./models_v2"
LOG_PATH = "./logs_v2"
CHECKPOINT_FREQ = 100000

# ==============================================================================
# 2. THE SIMULATION ENVIRONMENT (Upgraded for V2)
# ==============================================================================

# --- Soldier Properties ---
SOLDIER_RADIUS = 4
SOLDIER_SPEED = 60
SOLDIER_HEALTH = 100
SOLDIER_DAMAGE = 10
SOLDIER_ATTACK_RANGE_SQ = 40**2
SOLDIER_ATTACK_COOLDOWN = 1.0
SEPARATION_DISTANCE_SQ = 15**2 # For Boid-like separation
SEPARATION_STRENGTH = 0.5    # How strongly soldiers avoid each other

class WarSimEnv_V2(gym.Env):
    """V2 Environment with Grid-based Observation Space."""
    def __init__(self):
        super(WarSimEnv_V2, self).__init__()
        self.screen_width = 640
        self.screen_height = 480
        
        # --- NEW: Grid-based Observation Space ---
        # Two channels: one for friendly density, one for enemy density
        self.observation_space = spaces.Box(low=0, high=1, shape=(GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        
        # Actions are the same: 0=ATTACK_COM, 1=HOLD, 2=SPREAD_OUT
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        """Generates the 16x16 grid observation."""
        obs_grid = np.zeros((GRID_SIZE, GRID_SIZE, 2), dtype=np.float32)
        
        # Get indices of living soldiers
        blue_alive_indices = np.where((self.teams == 0) & (self.aliveness == 1))[0]
        red_alive_indices = np.where((self.teams == 1) & (self.aliveness == 1))[0]

        # Calculate grid cell size
        cell_w = self.screen_width / GRID_SIZE
        cell_h = self.screen_height / GRID_SIZE

        # Populate friendly density map (Channel 0)
        if len(blue_alive_indices) > 0:
            blue_positions = self.positions[blue_alive_indices]
            grid_x = np.floor(blue_positions[:, 0] / cell_w).astype(np.int32)
            grid_y = np.floor(blue_positions[:, 1] / cell_h).astype(np.int32)
            np.add.at(obs_grid[:, :, 0], (grid_y, grid_x), 1)

        # Populate enemy density map (Channel 1)
        if len(red_alive_indices) > 0:
            red_positions = self.positions[red_alive_indices]
            grid_x = np.floor(red_positions[:, 0] / cell_w).astype(np.int32)
            grid_y = np.floor(red_positions[:, 1] / cell_h).astype(np.int32)
            np.add.at(obs_grid[:, :, 1], (grid_y, grid_x), 1)
        
        # Normalize the density maps
        if np.max(obs_grid) > 0:
            obs_grid /= np.max(obs_grid)
            
        return obs_grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.obstacles = np.array([
            [random.uniform(self.screen_width / 4, self.screen_width * 3 / 4 - 80), random.uniform(0, self.screen_height - 80), random.randint(30, 80), random.randint(30, 80)]
            for _ in range(NUM_OBSTACLES)
        ], dtype=np.float32)

        total_soldiers = TROOP_COUNT * 2
        self.positions = np.zeros((total_soldiers, 2), dtype=np.float32)
        self.teams = np.zeros(total_soldiers, dtype=np.int8)
        self.healths = np.full(total_soldiers, SOLDIER_HEALTH, dtype=np.float32)
        self.aliveness = np.ones(total_soldiers, dtype=np.bool_)
        self.attack_cooldowns = np.zeros(total_soldiers, dtype=np.float32)
        self.targets = np.full(total_soldiers, -1, dtype=np.int32)

        for i in range(TROOP_COUNT):
            self.positions[i] = [random.uniform(0, self.screen_width / 4), random.uniform(0, self.screen_height)]
            self.teams[i] = 0 # Blue
            
            red_idx = i + TROOP_COUNT
            self.positions[red_idx] = [random.uniform(self.screen_width * 3/4, self.screen_width), random.uniform(0, self.screen_height)]
            self.teams[red_idx] = 1 # Red

        self.prev_blue_health = np.sum(self.healths[self.teams == 0])
        self.prev_red_health = np.sum(self.healths[self.teams == 1])

        return self._get_obs(), {}

    def step(self, action):
        self.positions, self.healths, self.aliveness, self.attack_cooldowns, self.targets = self.update_simulation(
            self.positions, self.healths, self.aliveness, self.attack_cooldowns, self.targets, self.teams, self.obstacles, action
        )
        self.current_step += 1
        
        current_blue_health = np.sum(self.healths[self.teams == 0])
        current_red_health = np.sum(self.healths[self.teams == 1])
        damage_dealt = (self.prev_red_health - current_red_health) * 0.1
        damage_taken = (self.prev_blue_health - current_blue_health) * 0.1
        blue_alive_count = np.sum(self.aliveness[self.teams == 0])
        survival_bonus = blue_alive_count * 0.001
        reward = (damage_dealt - damage_taken) + survival_bonus - 0.01
        
        self.prev_blue_health = current_blue_health
        self.prev_red_health = current_red_health
        
        observation = self._get_obs()
        
        blue_alive = blue_alive_count > 0
        red_alive = np.sum(self.aliveness[self.teams == 1]) > 0
        
        terminated = not blue_alive or not red_alive
        truncated = self.current_step >= MAX_STEPS_PER_EPISODE
        
        if terminated:
            if blue_alive: reward += 500
            else: reward -= 500

        return observation, reward, terminated, truncated, {}

    @staticmethod
    @numba.jit(nopython=True)
    def update_simulation(positions, healths, aliveness, attack_cooldowns, targets, teams, obstacles, blue_command):
        dt = 0.1
        num_soldiers = len(positions)
        for i in range(num_soldiers):
            if attack_cooldowns[i] > 0:
                attack_cooldowns[i] -= dt

        for i in range(num_soldiers):
            if not aliveness[i]: continue

            if targets[i] != -1 and not aliveness[targets[i]]:
                targets[i] = -1
            if targets[i] == -1 and np.random.rand() < 0.1:
                enemy_indices = np.where((teams != teams[i]) & (aliveness == 1))[0]
                if len(enemy_indices) > 0:
                    target_idx, _ = find_target_numba(positions[i], positions[enemy_indices], aliveness[enemy_indices])
                    targets[i] = enemy_indices[target_idx]

            move_direction = np.zeros(2, dtype=np.float32)
            target_idx = targets[i]
            if target_idx != -1:
                dx = positions[target_idx][0] - positions[i][0]
                dy = positions[target_idx][1] - positions[i][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq < SOLDIER_ATTACK_RANGE_SQ:
                    if attack_cooldowns[i] <= 0:
                        healths[target_idx] -= SOLDIER_DAMAGE
                        if healths[target_idx] <= 0: aliveness[target_idx] = False
                        attack_cooldowns[i] = SOLDIER_ATTACK_COOLDOWN
                else:
                    dist = np.sqrt(dist_sq)
                    move_direction[0] = dx / dist
                    move_direction[1] = dy / dist
            else:
                command = blue_command if teams[i] == 0 else 0
                if command == 0:
                    enemy_indices = np.where((teams != teams[i]) & (aliveness == 1))[0]
                    if len(enemy_indices) > 0:
                        enemy_com_x = np.mean(positions[enemy_indices, 0])
                        enemy_com_y = np.mean(positions[enemy_indices, 1])
                        dx = enemy_com_x - positions[i, 0]
                        dy = enemy_com_y - positions[i, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 1:
                            move_direction[0] = dx / dist
                            move_direction[1] = dy / dist
            
            # --- NEW: Boid Separation Logic ---
            separation_vec = np.zeros(2, dtype=np.float32)
            ally_indices = np.where((teams == teams[i]) & (aliveness == 1) & (np.arange(num_soldiers) != i))[0]
            if len(ally_indices) > 0:
                for j in ally_indices:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx*dx + dy*dy
                    if dist_sq < SEPARATION_DISTANCE_SQ and dist_sq > 0:
                        dist = np.sqrt(dist_sq)
                        separation_vec[0] += dx / dist
                        separation_vec[1] += dy / dist
            
            # Combine movement vectors
            final_move = move_direction + separation_vec * SEPARATION_STRENGTH
            if np.sum(np.abs(final_move)) > 0:
                 final_move = final_move / np.linalg.norm(final_move)


            if final_move[0] != 0 or final_move[1] != 0:
                next_x = positions[i, 0] + final_move[0] * SOLDIER_SPEED * dt
                next_y = positions[i, 1] + final_move[1] * SOLDIER_SPEED * dt
                collided = False
                for j in range(len(obstacles)):
                    obs = obstacles[j]
                    if obs[0] < next_x < obs[0] + obs[2] and obs[1] < next_y < obs[1] + obs[3]:
                        collided = True
                        break
                if not collided:
                    positions[i, 0] = next_x
                    positions[i, 1] = next_y

        return positions, healths, aliveness, attack_cooldowns, targets

# ==============================================================================
# 4. MAIN TRAINING EXECUTION (V2)
# ==============================================================================

def get_latest_checkpoint(path):
    if not os.path.isdir(path): return None
    files = [f for f in os.listdir(path) if f.startswith("warsim_v2_model_") and f.endswith(".zip")]
    if not files: return None
    steps = [int(re.search(r"(\d+)_steps", f).group(1)) for f in files]
    return os.path.join(path, f"warsim_v2_model_{max(steps)}_steps.zip")

if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    
    print("--- Initializing V2 Vectorized Environments (Grid Observation) ---")
    env = SubprocVecEnv([lambda: WarSimEnv_V2() for i in range(NUM_ENVIRONMENTS)])

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // NUM_ENVIRONMENTS, 1),
        save_path=MODEL_SAVE_PATH,
        name_prefix="warsim_v2_model"
    )

    latest_checkpoint = get_latest_checkpoint(MODEL_SAVE_PATH)
    
    # --- NEW: Using CnnPolicy for grid-based observations ---
    if latest_checkpoint:
        print(f"--- Resuming V2 training from {latest_checkpoint} ---")
        model = PPO.load(latest_checkpoint, env=env, device="cuda")
        completed_steps = int(re.search(r"(\d+)_steps", latest_checkpoint).group(1))
        remaining_timesteps = TOTAL_TIMESTEPS - completed_steps
    else:
        print("--- Starting a new V2 training run. ---")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=LOG_PATH,
            device="cuda"
        )
        remaining_timesteps = TOTAL_TIMESTEPS

    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False if latest_checkpoint else True
        )

    final_model_path = os.path.join(MODEL_SAVE_PATH, "warsim_v2_model_final")
    model.save(final_model_path)
    print(f"--- V2 Training Complete. Final model saved to {final_model_path} ---")
