"""
replay_paths.py  –  Visualise saved survivor/failure paths from sampled_paths.npz

Usage:
    python analysis/neural_nets/replay_paths.py \
        --npz  output/<run>/trial_1/gen_25/sampled_paths.npz \
        --map  environments/images/6.png \
        [--robot_radius 5] \
        [--scale 2] \
        [--fps 60] \
        [--robot_index 0]        # optional: replay only one robot (0-indexed into the npz)
        [--auto_advance goal]    # auto-advance mode: 'goal' (when goal reached), 'end' (at path end), or 'none' (default: 'goal')

Controls during playback:
    SPACE       pause / resume
    LEFT/RIGHT  step backward / forward (when paused)
    N           next robot
    P           previous robot
    A           toggle auto-advance mode (cycle: goal -> end -> none)
    G           jump to goal success step (if available)
    Q / ESC     quit
"""

import argparse
import os
import sys
import math
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOUR_BG          = (255, 255, 255)
COLOUR_TRAIL_SURV  = (0,   200,  80)   # green trail for survivors
COLOUR_TRAIL_FAIL  = (220,  60,  60)   # red trail for failures
COLOUR_ROBOT_SURV  = (0,   140, 255)   # blue robot body  (survivor)
COLOUR_ROBOT_FAIL  = (255, 140,   0)   # orange robot body (failure)
COLOUR_HEADING     = (255,   0,   0)   # heading arrow
COLOUR_GOAL        = (0,   255,   0)   # goal ring (bright green)
COLOUR_GOAL_INNER  = (100, 255, 100)   # inner circle (lighter green)
COLOUR_GOAL_CENTER = (255, 255, 255)   # center bullseye
COLOUR_TEXT_SURV   = (0,   180,  60)
COLOUR_TEXT_FAIL   = (200,  40,  40)
COLOUR_TEXT_UI     = (30,   30,  30)
COLOUR_AUTO_GOAL   = (0,   255, 128)   # teal for auto-advance on goal mode
COLOUR_AUTO_END    = (255, 128, 0)     # orange for auto-advance on end mode


def load_data(npz_path):
    """Load npz file with paths, labels, and optionally goal positions."""
    data = np.load(npz_path, allow_pickle=True)
    paths  = data["paths"]    # (steps, n_robots, 3)  x / y / orientation
    labels = data["labels"]   # (n_robots,)  "survivor" | "failure"
    indices = data["indices"] # original population indices
    
    # Try to load goal positions (may not be present in older files)
    try:
        goal_x = data["goal_x"]
        goal_y = data["goal_y"]
        has_goals = True
    except KeyError:
        print("Warning: No goal positions found in npz file. Goal won't be displayed.")
        goal_x = None
        goal_y = None
        has_goals = False
    
    # Try to load success_steps if available
    try:
        success_steps = data["success_steps"]
    except KeyError:
        success_steps = None
    
    return paths, labels, indices, goal_x, goal_y, success_steps, has_goals


def load_map(map_path, scale):
    """Load and scale map image."""
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load map: {map_path}")
    h, w = img.shape
    rgb = np.stack([img] * 3, axis=-1)          # (H, W, 3)  grayscale → RGB
    return rgb, w, h


def make_map_surface(pygame, rgb, scale):
    """Convert numpy map to a scaled pygame Surface."""
    h, w = rgb.shape[:2]
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(surf, (w * scale, h * scale))


def get_goal_success_step(paths, goal_x, goal_y, robot_idx, success_threshold=6.0):
    """
    Find the first step where the robot reaches the goal.
    Returns step index or None if never reached.
    """
    if goal_x is None or goal_y is None:
        return None
    
    for step in range(paths.shape[0]):
        dx = paths[step, robot_idx, 0] - goal_x[robot_idx]
        dy = paths[step, robot_idx, 1] - goal_y[robot_idx]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist <= success_threshold:
            return step
    return None


def draw_frame(screen, pygame, map_surf, paths, labels, goal_x, goal_y, success_steps,
               robot_idx, step, scale, robot_radius, font, show_trail=True, auto_mode="goal"):
    """Draw one frame for robot `robot_idx` at time `step`."""
    screen.blit(map_surf, (0, 0))

    label  = str(labels[robot_idx])
    is_surv = label == "survivor"
    trail_colour = COLOUR_TRAIL_SURV if is_surv else COLOUR_TRAIL_FAIL
    robot_colour = COLOUR_ROBOT_SURV if is_surv else COLOUR_ROBOT_FAIL
    text_colour  = COLOUR_TEXT_SURV  if is_surv else COLOUR_TEXT_FAIL

    # --- Goal marker (if goal coordinates available) ---
    if goal_x is not None and goal_y is not None:
        # Get goal position for this specific robot
        gx_scaled = int(goal_x[robot_idx] * scale)
        gy_scaled = int(goal_y[robot_idx] * scale)
        radius = int(robot_radius * scale * 1.5)
        
        # Check if goal is currently reached
        dx = paths[step, robot_idx, 0] - goal_x[robot_idx]
        dy = paths[step, robot_idx, 1] - goal_y[robot_idx]
        dist = math.sqrt(dx*dx + dy*dy)
        goal_reached = dist <= 6.0
        
        # Choose goal colour based on whether reached
        goal_colour = (0, 255, 0) if not goal_reached else (255, 215, 0)  # green normally, gold when reached
        
        # Outer ring
        pygame.draw.circle(screen, goal_colour, (gx_scaled, gy_scaled), radius, 3)
        # Inner circle - only if radius is large enough
        if radius >= 6:
            inner_colour = (100, 255, 100) if not goal_reached else (255, 235, 100)
            pygame.draw.circle(screen, inner_colour, (gx_scaled, gy_scaled), max(3, radius // 2))
            # Center bullseye
            centre_colour = (255, 255, 255) if not goal_reached else (255, 200, 50)
            pygame.draw.circle(screen, centre_colour, (gx_scaled, gy_scaled), max(2, radius // 4))
        else:
            pygame.draw.circle(screen, goal_colour, (gx_scaled, gy_scaled), max(2, radius // 2))

    x_hist = paths[:step + 1, robot_idx, 0] * scale
    y_hist = paths[:step + 1, robot_idx, 1] * scale

    # --- Trail ---
    if show_trail and len(x_hist) >= 2:
        pts = list(zip(x_hist.astype(int), y_hist.astype(int)))
        pygame.draw.lines(screen, trail_colour, False, pts, 2)

    # --- Robot body at current step ---
    cx = int(paths[step, robot_idx, 0] * scale)
    cy = int(paths[step, robot_idx, 1] * scale)
    r  = max(3, int(robot_radius * scale))
    pygame.draw.circle(screen, robot_colour, (cx, cy), r)

    # --- Heading arrow ---
    angle_rad = math.radians(paths[step, robot_idx, 2])
    arrow_len = r * 1.8
    ex = int(cx + arrow_len * math.cos(angle_rad))
    ey = int(cy + arrow_len * math.sin(angle_rad))
    pygame.draw.line(screen, COLOUR_HEADING, (cx, cy), (ex, ey), 2)
    pygame.draw.circle(screen, (255, 255, 0), (ex, ey), 2)

    # --- HUD ---
    total_steps = paths.shape[0]
    n_robots    = paths.shape[1]
    
    # Distance to goal (if available)
    dist_to_goal_str = ""
    if goal_x is not None and goal_y is not None:
        dx = paths[step, robot_idx, 0] - goal_x[robot_idx]
        dy = paths[step, robot_idx, 1] - goal_y[robot_idx]
        dist = math.sqrt(dx*dx + dy*dy)
        dist_to_goal_str = f" | dist to goal: {dist:.1f}"
        
        if dist <= 6.0:
            dist_to_goal_str += " ✓ GOAL REACHED!"
    
    # Success info (if available)
    success_str = ""
    if success_steps is not None and success_steps[robot_idx] > 0:
        success_str = f" | SUCCESS at step {int(success_steps[robot_idx])}"
    
    # Auto-advance mode indicator
    auto_mode_str = ""
    if auto_mode == "goal":
        auto_mode_str = " [AUTO: GOAL]"
        auto_mode_colour = COLOUR_AUTO_GOAL
    elif auto_mode == "end":
        auto_mode_str = " [AUTO: END]"
        auto_mode_colour = COLOUR_AUTO_END
    else:
        auto_mode_str = " [AUTO: OFF]"
        auto_mode_colour = COLOUR_TEXT_UI
    
    # Add goal success step if available
    goal_step_str = ""
    if hasattr(replay, '_goal_step_cache') and robot_idx in replay._goal_step_cache:
        gs = replay._goal_step_cache[robot_idx]
        if gs is not None:
            goal_step_str = f" | goal at step {gs}"
            if step >= gs:
                goal_step_str += " [PAST GOAL]"
    
    hud_lines = [
        f"Robot {robot_idx + 1}/{n_robots}  |  pop index: {robot_idx}",
        f"Label: {label.upper()}{dist_to_goal_str}{success_str}{goal_step_str}",
        f"Step: {step + 1}/{total_steps}",
        f"SPACE=pause  N/P=next/prev  A=toggle auto{auto_mode_str}  G=jump to goal  Q=quit",
    ]
    for i, line in enumerate(hud_lines):
        if i == 3 and "AUTO:" in line:
            # Highlight auto-advance mode indicator
            parts = line.split(auto_mode_str)
            surf1 = font.render(parts[0], True, COLOUR_TEXT_UI)
            surf2 = font.render(auto_mode_str, True, auto_mode_colour)
            screen.blit(surf1, (8, 8 + i * 20))
            screen.blit(surf2, (8 + surf1.get_width(), 8 + i * 20))
            if len(parts) > 1:
                surf3 = font.render(parts[1], True, COLOUR_TEXT_UI)
                screen.blit(surf3, (8 + surf1.get_width() + surf2.get_width(), 8 + i * 20))
        else:
            colour = text_colour if i == 1 else COLOUR_TEXT_UI
            surf = font.render(line, True, colour)
            screen.blit(surf, (8, 8 + i * 20))

    pygame.display.flip()


def replay(npz_path, map_path, robot_radius=5, scale=2, fps=60,
           robot_index=None, auto_advance="goal"):
    import pygame

    paths, labels, indices, goal_x, goal_y, success_steps, has_goals = load_data(npz_path)
    rgb, map_w, map_h = load_map(map_path, scale)

    total_steps = paths.shape[0]
    n_robots    = paths.shape[1]

    pygame.init()
    win_w = map_w * scale
    win_h = map_h * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Path Replay" + (" (with goal)" if has_goals else ""))
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 14)

    map_surf = make_map_surface(pygame, rgb, scale)

    # Pre-compute goal success steps for all robots
    goal_steps = {}
    if has_goals:
        for i in range(n_robots):
            goal_steps[i] = get_goal_success_step(paths, goal_x, goal_y, i)
    
    # Store in replay function for access in draw_frame
    replay._goal_step_cache = goal_steps

    # Which robots to show
    robot_order = list(range(n_robots))
    if robot_index is not None:
        if robot_index >= n_robots:
            print(f"robot_index {robot_index} out of range (0–{n_robots-1}), showing all.")
        else:
            robot_order = [robot_index]

    ri       = 0          # index into robot_order
    step     = 0
    paused   = False
    running  = True
    auto_mode = auto_advance  # 'goal', 'end', or 'none'

    # Print info about loaded data
    print(f"\n=== Replay Info ===")
    print(f"Total steps: {total_steps}")
    print(f"Number of robots: {n_robots}")
    print(f"Labels: {dict(zip(range(len(labels)), labels))}")
    if has_goals:
        print(f"Goal positions available (will be displayed as green rings)")
        for i, gs in goal_steps.items():
            if gs is not None:
                print(f"  Robot {i} ({labels[i]}): goal reached at step {gs}")
            else:
                print(f"  Robot {i} ({labels[i]}): goal NOT reached")
    else:
        print(f"No goal positions in file (only paths)")
    print(f"Auto-advance mode: {auto_mode}")
    print("==================\n")

    while running:
        robot_idx = robot_order[ri]
        goal_step = goal_steps.get(robot_idx) if has_goals else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_a:
                    # Cycle auto-advance modes
                    if auto_mode == "goal":
                        auto_mode = "end"
                    elif auto_mode == "end":
                        auto_mode = "none"
                    else:
                        auto_mode = "goal"
                    print(f"Auto-advance mode: {auto_mode}")
                elif event.key == pygame.K_n:
                    ri   = (ri + 1) % len(robot_order)
                    step = 0
                    paused = False
                elif event.key == pygame.K_p:
                    ri   = (ri - 1) % len(robot_order)
                    step = 0
                    paused = False
                elif event.key == pygame.K_g and goal_step is not None:
                    # Jump to goal success step
                    step = min(goal_step, total_steps - 1)
                    if auto_mode == "goal":
                        # Auto-advance after showing goal for a moment
                        pass
                elif event.key == pygame.K_RIGHT and paused:
                    step = min(step + 1, total_steps - 1)
                elif event.key == pygame.K_LEFT and paused:
                    step = max(step - 1, 0)

        draw_frame(screen, pygame, map_surf, paths, labels, 
                   goal_x, goal_y, success_steps,
                   robot_idx, step, scale, robot_radius, font, auto_mode=auto_mode)

        if not paused:
            step += 1
            
            # Check for auto-advance conditions
            advance = False
            
            if auto_mode == "goal" and goal_step is not None:
                # Advance when goal is reached
                if step > goal_step:
                    advance = True
            elif auto_mode == "end":
                # Advance at end of path
                if step >= total_steps:
                    advance = True
            
            if advance:
                step = 0
                ri = (ri + 1) % len(robot_order)
                # Print when switching robots
                new_robot = robot_order[ri]
                new_goal_step = goal_steps.get(new_robot) if has_goals else None
                if new_goal_step is not None:
                    print(f"Switching to robot {new_robot}: {labels[new_robot]} (goal at step {new_goal_step})")
                else:
                    print(f"Switching to robot {new_robot}: {labels[new_robot]}")
                
                # Reset pause state when auto-advancing
                if auto_mode != "none":
                    paused = False

        clock.tick(fps)

    pygame.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay saved robot paths with auto-advance on goal.")
    parser.add_argument("--npz",          required=True,  help="Path to sampled_paths.npz")
    parser.add_argument("--map",          required=True,  help="Path to map image (e.g. environments/images/6.png)")
    parser.add_argument("--robot_radius", type=int,   default=5,  help="Robot radius in map pixels (default: 5)")
    parser.add_argument("--scale",        type=int,   default=2,  help="Display scale factor (default: 2)")
    parser.add_argument("--fps",          type=int,   default=60, help="Playback FPS (default: 60)")
    parser.add_argument("--robot_index",  type=int,   default=None, help="Show only this robot (0-indexed). Omit to cycle all.")
    parser.add_argument("--auto_advance", type=str,   default="goal", 
                        choices=["goal", "end", "none"],
                        help="Auto-advance mode: 'goal' (when goal reached), 'end' (at path end), or 'none' (default: 'goal')")
    args = parser.parse_args()

    replay(
        npz_path     = args.npz,
        map_path     = args.map,
        robot_radius = args.robot_radius,
        scale        = args.scale,
        fps          = args.fps,
        robot_index  = args.robot_index,
        auto_advance = args.auto_advance,
    )