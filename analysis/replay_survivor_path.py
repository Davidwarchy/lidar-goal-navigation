"""
replay_paths.py  –  Visualise saved survivor/failure paths from sampled_paths.npz

Usage:
    python analysis/neural_nets/replay_paths.py \\
        --npz  output/<run>/trial_1/gen_25/sampled_paths.npz \\
        --map  environments/images/6.png \\
        [--robot_radius 5] \\
        [--scale 2] \\
        [--fps 60] \\
        [--robot_index 0]        # optional: replay only one robot (0-indexed into the npz)

Controls during playback:
    SPACE       pause / resume
    LEFT/RIGHT  step backward / forward (when paused)
    N           next robot
    P           previous robot
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
COLOUR_GOAL        = (0,   255,   0)   # goal ring
COLOUR_TEXT_SURV   = (0,   180,  60)
COLOUR_TEXT_FAIL   = (200,  40,  40)
COLOUR_TEXT_UI     = (30,   30,  30)


def load_data(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    paths  = data["paths"]    # (steps, n_robots, 3)  x / y / orientation
    labels = data["labels"]   # (n_robots,)  "survivor" | "failure"
    indices = data["indices"] # original population indices
    return paths, labels, indices


def load_map(map_path, scale):
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load map: {map_path}")
    h, w = img.shape
    rgb = np.stack([img] * 3, axis=-1)          # (H, W, 3)  grayscale → RGB
    # pygame expects (W, H, 3) with surfarray
    return rgb, w, h


def make_map_surface(pygame, rgb, scale):
    """Convert numpy map to a scaled pygame Surface."""
    h, w = rgb.shape[:2]
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(surf, (w * scale, h * scale))


def draw_frame(screen, pygame, map_surf, paths, labels, robot_idx, step,
               scale, robot_radius, font, show_trail=True):
    """Draw one frame for robot `robot_idx` at time `step`."""
    screen.blit(map_surf, (0, 0))

    label  = str(labels[robot_idx])
    is_surv = label == "survivor"
    trail_colour = COLOUR_TRAIL_SURV if is_surv else COLOUR_TRAIL_FAIL
    robot_colour = COLOUR_ROBOT_SURV if is_surv else COLOUR_ROBOT_FAIL
    text_colour  = COLOUR_TEXT_SURV  if is_surv else COLOUR_TEXT_FAIL

    x_hist = paths[:step + 1, robot_idx, 0] * scale
    y_hist = paths[:step + 1, robot_idx, 1] * scale

    # --- Trail ---
    if show_trail and len(x_hist) >= 2:
        pts = list(zip(x_hist.astype(int), y_hist.astype(int)))
        pygame.draw.lines(screen, trail_colour, False, pts, 2)

    # --- Goal marker (use start position of robot as reference —
    #     goal coords aren't stored in the npz, so we mark the final
    #     position of survivors as a proxy, or skip for failures) ---
    # NOTE: if you want to save goal_x/goal_y into the npz, add them
    #       in the save block in _run_single_trial and draw a circle here.

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
    hud_lines = [
        f"Robot {robot_idx + 1}/{n_robots}  |  pop index: {robot_idx}",
        f"Label: {label.upper()}",
        f"Step: {step + 1}/{total_steps}",
        "SPACE=pause  N/P=next/prev  Q=quit",
    ]
    for i, line in enumerate(hud_lines):
        colour = text_colour if i == 1 else COLOUR_TEXT_UI
        surf = font.render(line, True, colour)
        screen.blit(surf, (8, 8 + i * 20))

    pygame.display.flip()


def replay(npz_path, map_path, robot_radius=5, scale=2, fps=60,
           robot_index=None):
    import pygame

    paths, labels, indices = load_data(npz_path)
    rgb, map_w, map_h = load_map(map_path, scale)

    total_steps = paths.shape[0]
    n_robots    = paths.shape[1]

    pygame.init()
    win_w = map_w * scale
    win_h = map_h * scale
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Path Replay")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 14)

    map_surf = make_map_surface(pygame, rgb, scale)

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

    while running:
        robot_idx = robot_order[ri]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    ri   = (ri + 1) % len(robot_order)
                    step = 0
                elif event.key == pygame.K_p:
                    ri   = (ri - 1) % len(robot_order)
                    step = 0
                elif event.key == pygame.K_RIGHT and paused:
                    step = min(step + 1, total_steps - 1)
                elif event.key == pygame.K_LEFT and paused:
                    step = max(step - 1, 0)

        draw_frame(screen, pygame, map_surf, paths, labels,
                   robot_idx, step, scale, robot_radius, font)

        if not paused:
            step += 1
            if step >= total_steps:
                # Auto-advance to next robot at end of path
                step = 0
                ri = (ri + 1) % len(robot_order)

        clock.tick(fps)

    pygame.quit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay saved robot paths.")
    parser.add_argument("--npz",          required=True,  help="Path to sampled_paths.npz")
    parser.add_argument("--map",          required=True,  help="Path to map image (e.g. environments/images/6.png)")
    parser.add_argument("--robot_radius", type=int,   default=5,  help="Robot radius in map pixels (default: 5)")
    parser.add_argument("--scale",        type=int,   default=2,  help="Display scale factor (default: 2)")
    parser.add_argument("--fps",          type=int,   default=60, help="Playback FPS (default: 60)")
    parser.add_argument("--robot_index",  type=int,   default=None, help="Show only this robot (0-indexed). Omit to cycle all.")
    args = parser.parse_args()

    replay(
        npz_path     = args.npz,
        map_path     = args.map,
        robot_radius = args.robot_radius,
        scale        = args.scale,
        fps          = args.fps,
        robot_index  = args.robot_index,
    )