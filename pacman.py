import math
import random
import sys
import pygame
from collections import deque
from dataclasses import dataclass

from ghosts import GhostAgent  


TILE = 32
FPS = 15
MAX_GAME_STEPS = 3000
GHOST_MINIMAX_DEPTH = 1
PACMAN_REPLAN_STEPS = 3

BLACK = (0, 0, 0)
GRAY = (40, 40, 48)
WHITE = (240, 240, 240)
YELLOW = (255, 214, 10)
RED = (255, 64, 64)
BLUE = (72, 132, 255)
GREEN = (80, 200, 120)
ORANGE = (255, 160, 72)
WALL_COL = (30, 30, 60)
PELLET_COL = (230, 230, 230)
POWER_COL = (255, 120, 200)
PATH_COL = (120, 200, 255)


def generate_random_maze(width=19, height=19):
    maze = [['#' for _ in range(width)] for _ in range(height)]

    def carve_passages(x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)
        maze[y][x] = '.'
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and maze[ny][nx] == '#'):
                maze[y + dy // 2][x + dx // 2] = '.'
                carve_passages(nx, ny)

    start_points = [(width // 2, height // 2)]
    if width > 6 and height > 6:
        start_points.extend([(3, 3), (width - 4, 3), (3, height - 4), (width - 4, height - 4)])

    for sx, sy in start_points:
        if maze[sy][sx] == '#':
            carve_passages(sx, sy)

    for _ in range(width * height // 4):
        x = random.randrange(1, width - 1)
        y = random.randrange(1, height - 1)
        if maze[y][x] == '#':
            nb = sum(1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                     if 0 <= x+dx < width and 0 <= y+dy < height and maze[y+dy][x+dx] == '.')
            if nb >= 1:
                maze[y][x] = '.'

    for y in [height//4, height//2, 3*height//4]:
        if 0 < y < height-1:
            for x in range(2, width-2, 2):
                if maze[y][x] == '#':
                    maze[y][x] = '.'
                    for dy in [-1,1]:
                        if 0 <= y+dy < height and maze[y+dy][x] == '.':
                            break

    for x in [width//4, width//2, 3*width//4]:
        if 0 < x < width-1:
            for y in range(2, height-2, 2):
                if maze[y][x] == '#':
                    maze[y][x] = '.'
                    for dx in [-1,1]:
                        if 0 <= x+dx < width and maze[y][x+dx] == '.':
                            break

    power_spots = [(1,1), (1,height-2), (width-2,1), (width-2,height-2)]
    for x, y in power_spots:
        if maze[y][x] == '.':
            maze[y][x] = 'o'

    empty_spots = [(x,y) for y in range(1,height-1) for x in range(1,width-1)
                   if maze[y][x] == '.' and (x,y) not in power_spots]
    for _ in range(6):
        if empty_spots:
            x, y = random.choice(empty_spots)
            maze[y][x] = 'o'
            empty_spots.remove((x,y))

    strategic_spots = [(width//2 + 2, height//2), (width//2 - 2, height//2),
                       (width//2, height//2 + 2), (width//2, height//2 - 2)]
    for x, y in strategic_spots:
        if (0 < x < width-1 and 0 < y < height-1 and maze[y][x] == '.' and (x,y) not in power_spots):
            maze[y][x] = 'o'
            if (x,y) in empty_spots:
                empty_spots.remove((x,y))

    center_y = height // 2
    center_x = width // 2
    all_spots = [(x,y) for y in range(1, height-1) for x in range(1, width-1) if maze[y][x] == '.']

    if len(all_spots) < 3:
        for y in range(max(1, center_y-2), min(height-1, center_y+3)):
            for x in range(max(1, center_x-2), min(width-1, center_x+3)):
                if maze[y][x] == '#':
                    maze[y][x] = '.'
                    all_spots.append((x, y))
                if len(all_spots) >= 3:
                    break
            if len(all_spots) >= 3:
                break

    if all_spots:
        pacman_candidates = []
        for x, y in all_spots:
            nb = sum(1 for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]
                     if 0 <= x+dx < width and 0 <= y+dy < height and maze[y+dy][x+dx] == '.')
            if nb >= 3:
                pacman_candidates.append((x, y))
        pac_pos = random.choice(pacman_candidates if pacman_candidates else all_spots)
        maze[pac_pos[1]][pac_pos[0]] = 'P'
        all_spots.remove(pac_pos)

        ghost_candidates = [pos for pos in all_spots if abs(pos[0]-pac_pos[0]) + abs(pos[1]-pac_pos[1]) >= 5]
        if len(ghost_candidates) >= 2:
            ghost_pos = random.sample(ghost_candidates, 2)
        elif len(all_spots) >= 2:
            ghost_pos = random.sample(all_spots, 2)
        else:
            center_spots = [(x,y) for y in range(center_y-2, center_y+3)
                            for x in range(center_x-2, center_x+3)
                            if maze[y][x] == '.' and (x,y) != pac_pos]
            ghost_pos = random.sample(center_spots, 2) if len(center_spots) >= 2 else all_spots[:2]

        if len(ghost_pos) >= 1:
            maze[ghost_pos[0][1]][ghost_pos[0][0]] = 'R'
        if len(ghost_pos) >= 2:
            maze[ghost_pos[1][1]][ghost_pos[1][0]] = 'B'

    return [''.join(row) for row in maze]

LEVEL = generate_random_maze(19, 19)
H = len(LEVEL)
W = len(LEVEL[0])

DIRS = [(1,0),(-1,0),(0,1),(0,-1)]

def in_bounds(x, y):
    return 0 <= x < W and 0 <= y < H

def neighbors(x, y):
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        if in_bounds(nx, ny) and LEVEL[ny][nx] != '#':
            yield nx, ny

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



@dataclass
class State:
    pac: tuple
    ghosts: tuple
    pellets: frozenset
    power: frozenset
    scared_timer: int

    def copy(self):
        return State(self.pac, tuple(self.ghosts), frozenset(self.pellets), frozenset(self.power), self.scared_timer)

pellets_set = set()
power_set = set()
pac_spawn = None
ghost_spawns = []
for y, row in enumerate(LEVEL):
    for x, ch in enumerate(row):
        if ch == '.':
            pellets_set.add((x, y))
        elif ch == 'o':
            power_set.add((x, y))
        elif ch == 'P':
            pac_spawn = (x, y)
        elif ch in ('R', 'B'):
            ghost_spawns.append((x, y))

INIT_STATE = State(pac_spawn, tuple(ghost_spawns), frozenset(pellets_set), frozenset(power_set), scared_timer=0)



class PacmanFuzzy:
    def mu_close(self, d):
        if d <= 6: return 1.0
        if d >= 12: return 0.0
        return (12 - d) / 6.0

    def mu_far(self, d):
        if d <= 8: return 0.0
        if d >= 18: return 1.0
        return (d - 8) / 10.0

    def mu_near_power(self, d):
        if d <= 3: return 1.0
        if d >= 8: return 0.0
        return (8 - d) / 5.0

    def mu_low_food(self, ratio):
        if ratio <= 0.1: return 1.0
        if ratio >= 0.3: return 0.0
        return (0.3 - ratio) / 0.2

    def decide(self, state: State, init_pellet_count: int):
        pg = min((manhattan(state.pac, g) for g in state.ghosts), default=999)
        pp = min(manhattan(state.pac, p) for p in state.power) if state.power else 999
        ratio = len(state.pellets) / max(1, init_pellet_count)
        pellets_left = len(state.pellets)

        close_g = self.mu_close(pg)
        far_g = self.mu_far(pg)
        near_pow = self.mu_near_power(pp)
        low_food = self.mu_low_food(ratio)

        r_power1 = min(close_g, near_pow) * 1.5
        r_power2 = near_pow * 0.8
        r_evade1 = close_g * (0.3 if pellets_left <= 10 else 0.8)
        r_food1 = min(far_g, low_food) * (3.0 if pellets_left <= 6 else 1.5)
        r_food2 = min(far_g, 1 - near_pow)
        r_food3 = 0.3
        r_evade2 = 0.4

        out_power = max(r_power1, r_power2)
        out_evade = max(r_evade1, r_evade2)
        out_food = max(r_food1, r_food2, r_food3)

        s = out_power + out_evade + out_food
        if s == 0:
            return 'EAT_FOOD'
        out_power /= s; out_evade /= s; out_food /= s
        scores = {'EAT_POWER': out_power, 'EVADE': out_evade, 'EAT_FOOD': out_food}
        return max(scores, key=lambda k: (scores[k], random.random()*0.01))


class Pathfinder:
    def astar(self, start, goals, walls_block=None, recent_path=None):
        if not goals:
            return []
        open_set = [(self.h(start, goals, recent_path), 0, start, None)]
        came = {}
        g = {start: 0}
        closed = set()

        while open_set:
            open_set.sort(key=lambda x: x[0])
            f, gcost, node, parent = open_set.pop(0)
            if node in closed:
                continue
            came[node] = parent
            closed.add(node)
            if node in goals:
                return self._reconstruct(came, node)
            for nx, ny in neighbors(*node):
                if walls_block and (nx, ny) in walls_block:
                    continue
                ng = gcost + 1
                if ng < g.get((nx, ny), 1e9):
                    g[(nx, ny)] = ng
                    nf = ng + self.h((nx, ny), goals, recent_path)
                    open_set.append((nf, ng, (nx, ny), node))
        return []

    def gbfs(self, start, goals, recent_path=None):
        if not goals:
            return []
        open_set = [(self.h(start, goals, recent_path), start, None)]
        closed = set()
        came = {}
        while open_set:
            open_set.sort(key=lambda x: x[0])
            h, node, parent = open_set.pop(0)
            if node in closed:
                continue
            came[node] = parent
            closed.add(node)
            if node in goals:
                return self._reconstruct(came, node)
            for nx, ny in neighbors(*node):
                if (nx, ny) not in closed:
                    open_set.append((self.h((nx, ny), goals, recent_path), (nx, ny), node))
        return []

    def _reconstruct(self, came, node):
        path = [node]
        while came[node] is not None:
            node = came[node]
            path.append(node)
        path.reverse()
        return path

    def h(self, node, goals, recent_path=None):
        if not goals:
            return 0
        min_dist = min(manhattan(node, g) for g in goals)
        if len(goals) <= 6:
            avg_dist = sum(manhattan(node, g) for g in goals) / len(goals)
            return 0.7 * min_dist + 0.3 * avg_dist
        path_penalty = 5 if (recent_path and node in recent_path) else 0
        return min_dist + path_penalty



class Renderer:
    def __init__(self, screen):
        self.screen = screen
        screen_height = screen.get_height()
        base_font_size = max(16, screen_height // 50)
        big_font_size = max(20, screen_height // 40)

        self.font = pygame.font.SysFont("consolas", base_font_size)
        self.big = pygame.font.SysFont("consolas", big_font_size, bold=True)
        self.huge = pygame.font.SysFont("consolas", big_font_size + 10, bold=True)


    def draw_game_over(self, won, score, steps, game_instance=None):
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()


        overlay = pygame.Surface((screen_width, screen_height))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(180)
        self.screen.blit(overlay, (0, 0))


        title_size = max(32, screen_height // 20)
        title_font = pygame.font.SysFont("consolas", title_size, bold=True)
        title_text = "PACMAN WINS!" if won else "GHOSTS WIN!"
        title_color = (255, 214, 10) if won else (255, 64, 64)
        title = title_font.render(title_text, True, title_color)

        stats_size = max(16, screen_height // 40)
        stats_font = pygame.font.SysFont("consolas", stats_size)
        score_text = stats_font.render(f"Final Score: {score:,}", True, (255, 214, 10))
        steps_text = stats_font.render(f"Steps: {steps}/{MAX_GAME_STEPS}", True, (240, 240, 240))
        hint_text = stats_font.render("Press SPACE to close game", True, (160, 160, 160))

        line_y = screen_height // 2
        pygame.draw.line(self.screen, title_color, (screen_width//4, line_y), (3*screen_width//4, line_y), 3)


        title_rect = title.get_rect(center=(screen_width//2, screen_height//2 - 80))
        score_rect = score_text.get_rect(center=(screen_width//2, screen_height//2 + 40))
        steps_rect = steps_text.get_rect(center=(screen_width//2, screen_height//2 + 80))
        hint_rect = hint_text.get_rect(center=(screen_width//2, screen_height//2 + 140))


        self.screen.blit(title, title_rect)
        self.screen.blit(score_text, score_rect)
        self.screen.blit(steps_text, steps_rect)
        self.screen.blit(hint_text, hint_rect)


    def draw_grid(self, state: State, path_overlay=None, game_instance=None):
        self.screen.fill(BLACK)
        if game_instance is None:
            tile_size = TILE
            offset_x, offset_y = 0, 0
        else:
            tile_size = game_instance.tile_size
            offset_x = game_instance.game_offset_x
            offset_y = game_instance.game_offset_y

        game_rect = (offset_x, offset_y, game_instance.game_width if game_instance else W*tile_size,
                     game_instance.game_height if game_instance else H*tile_size)
        pygame.draw.rect(self.screen, GRAY, game_rect)

        for y, row in enumerate(LEVEL):
            for x, ch in enumerate(row):
                rx, ry = offset_x + x*tile_size, offset_y + y*tile_size
                if ch == '#':
                    pygame.draw.rect(self.screen, WALL_COL, (rx, ry, tile_size, tile_size))
                else:
                    pygame.draw.rect(self.screen, (16,16,22), (rx, ry, tile_size, tile_size))
                    if (x,y) in state.pellets:
                        pellet_size = max(2, tile_size // 8)
                        pygame.draw.circle(self.screen, PELLET_COL, (rx+tile_size//2, ry+tile_size//2), pellet_size)
                    if (x,y) in state.power:
                        power_size = max(4, tile_size // 4)
                        pygame.draw.circle(self.screen, POWER_COL, (rx+tile_size//2, ry+tile_size//2), power_size, width=2)

        if path_overlay and len(path_overlay) > 1:
            pts = [(offset_x + x*tile_size+tile_size//2, offset_y + y*tile_size+tile_size//2) for (x,y) in path_overlay]
            pygame.draw.lines(self.screen, PATH_COL, False, pts, max(1, tile_size // 12))

        px, py = state.pac
        center_x = offset_x + px*tile_size+tile_size//2
        center_y = offset_y + py*tile_size+tile_size//2
        color = YELLOW if state.scared_timer == 0 else ORANGE
        pac_size = max(6, tile_size//2 - 2)

        if game_instance:
            self.draw_animated_pacman(center_x, center_y, pac_size, color,
                                      game_instance.pacman_direction, game_instance.animation_frame)
        else:
            pygame.draw.circle(self.screen, color, (center_x, center_y), pac_size)

        for i, g in enumerate(state.ghosts):
            gx, gy = g
            center_x = offset_x + gx*tile_size+tile_size//2
            center_y = offset_y + gy*tile_size+tile_size//2
            col = RED if i == 0 else BLUE
            if state.scared_timer > 0:
                col = GREEN
            ghost_size = max(6, tile_size//2 - 2)
            if game_instance:
                direction = game_instance.ghost_directions[i] if i < len(game_instance.ghost_directions) else (0, 0)
                self.draw_animated_ghost(center_x, center_y, ghost_size, col,
                                         direction, game_instance.animation_frame, state.scared_timer > 0)
            else:
                pygame.draw.circle(self.screen, col, (center_x, center_y), ghost_size)

    def draw_hud(self, steps, mode, score, lives=3, ghost_behaviors=None, game_instance=None):
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        hud_y = (game_instance.game_offset_y + game_instance.game_height + 20) if game_instance else (screen_height - 150)

        hud_rect = (0, hud_y - 10, screen_width, 150)
        pygame.draw.rect(self.screen, (20, 20, 30), hud_rect)

        section_width = screen_width // 4
        s1 = self.big.render(f"Steps: {steps}/{MAX_GAME_STEPS}", True, WHITE)
        s2 = self.big.render(f"Pacman: {mode}", True, WHITE)
        s3 = self.big.render(f"Score: {score:,}/30000", True, YELLOW)
        lives_text = "♥" * lives
        s4 = self.big.render(f"Lives: {lives_text}", True, RED)

        y_pos1 = hud_y + 10
        x1 = (section_width - s1.get_width()) // 2
        x2 = section_width + (section_width - s2.get_width()) // 2
        x3 = 2 * section_width + (section_width - s3.get_width()) // 2
        x4 = 3 * section_width + (section_width - s4.get_width()) // 2

        self.screen.blit(s1, (x1, y_pos1))
        self.screen.blit(s2, (x2, y_pos1))
        self.screen.blit(s3, (x3, y_pos1))
        self.screen.blit(s4, (x4, y_pos1))

        if ghost_behaviors:
            y_pos2 = hud_y + 60
            behavior_abbrev = {
                'AGGRESSIVE_PURSUIT': 'AGGRESSIVE',
                'COORDINATED_HUNT': 'COORDINATED',
                'INTERCEPT': 'INTERCEPT',
                'SCATTER': 'SCATTER',
                'DEFENSIVE': 'DEFENSIVE'
            }
            g1_text = behavior_abbrev.get(ghost_behaviors.get(0, 'UNKNOWN'), 'UNKNOWN')
            g2_text = behavior_abbrev.get(ghost_behaviors.get(1, 'UNKNOWN'), 'UNKNOWN')
            g1 = self.font.render(f"Red Ghost: {g1_text}", True, RED)
            g2 = self.font.render(f"Blue Ghost: {g2_text}", True, BLUE)
            x_g1 = (screen_width // 2 - g1.get_width()) // 2
            x_g2 = screen_width // 2 + (screen_width // 2 - g2.get_width()) // 2
            self.screen.blit(g1, (x_g1, y_pos2))
            self.screen.blit(g2, (x_g2, y_pos2))

        instructions = self.font.render("Press ESC to exit fullscreen", True, (128, 128, 128))
        self.screen.blit(instructions, (10, screen_height - 30))

    def draw_animated_pacman(self, cx, cy, size, color, direction, frame):
        mouth_cycle = (frame // 4) % 4
        mouth_open = mouth_cycle < 2
        if not mouth_open:
            pygame.draw.circle(self.screen, color, (int(cx), int(cy)), size)
        else:
            dx, dy = direction
            if dx > 0:
                start_angle, end_angle = 0.4, 5.9
            elif dx < 0:
                start_angle, end_angle = 3.54, 2.74
            elif dy > 0:
                start_angle, end_angle = 1.97, 1.17
            elif dy < 0:
                start_angle, end_angle = 4.51, 5.31
            else:
                start_angle, end_angle = 0.4, 5.9
            rect = (int(cx - size), int(cy - size), size * 2, size * 2)
            pygame.draw.arc(self.screen, color, rect, start_angle, end_angle, size)
            points = [(cx, cy)]
            for angle in [start_angle, end_angle]:
                x = cx + size * math.cos(angle)
                y = cy + size * math.sin(angle)
                points.append((x, y))
            if len(points) >= 3:
                pygame.draw.polygon(self.screen, color, points)

    def draw_animated_ghost(self, cx, cy, size, color, direction, frame, scared):
        bob_offset = int(2 * math.sin(frame * 0.3))
        ay = cy + bob_offset
        if scared:
            flash_cycle = (frame // 8) % 2
            color = (0, 0, 255) if flash_cycle == 0 else (255, 255, 255)

        pygame.draw.circle(self.screen, color, (int(cx), int(ay - size//4)), size)

        bottom_points = []
        wave_width = size // 3
        num_waves = 3
        left_x = cx - size
        right_x = cx + size
        bottom_y = ay + size//2
        for i in range(num_waves + 1):
            x = left_x + (right_x - left_x) * i / num_waves
            wave_offset = int(wave_width * math.sin(frame * 0.2 + i * 2))
            y = bottom_y + wave_offset
            bottom_points.append((x, y))
        ghost_points = [(left_x, ay - size//2)]
        ghost_points.extend(bottom_points)
        ghost_points.append((right_x, ay - size//2))
        if len(ghost_points) >= 3:
            pygame.draw.polygon(self.screen, color, ghost_points)

        eye_size = max(2, size // 6)
        eye_offset_x = size // 3
        eye_offset_y = size // 4
        dx, dy = direction
        eye_shift_x = int(dx * 2)
        eye_shift_y = int(dy * 2)
        lx = int(cx - eye_offset_x + eye_shift_x)
        ly = int(ay - eye_offset_y + eye_shift_y)
        rx = int(cx + eye_offset_x + eye_shift_x)
        ry = int(ay - eye_offset_y + eye_shift_y)
        pygame.draw.circle(self.screen, (255, 255, 255), (lx, ly), eye_size)
        pygame.draw.circle(self.screen, (0, 0, 0), (lx, ly), max(1, eye_size//2))
        pygame.draw.circle(self.screen, (255, 255, 255), (rx, ry), eye_size)
        pygame.draw.circle(self.screen, (0, 0, 0), (rx, ry), max(1, eye_size//2))



class PacmanAgent:
    def __init__(self, init_pellet_count):
        self.fuzzy = PacmanFuzzy()
        self.pathfinder = Pathfinder()
        self.path_cache = []
        self.mode = 'EAT_FOOD'
        self.replan_cooldown = 0
        self.init_pellets = init_pellet_count
        self.eaten_pellets_history = deque(maxlen=20)
        self.recent_path = deque(maxlen=15)
        self.position_history = deque(maxlen=25)
        self.stuck_counter = 0
        self.last_position = None
        self.emergency_mode = False
        self.emergency_cooldown = 0

    def choose_action(self, state: State):
        current_pos = state.pac
        if self.last_position == current_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.position_history.append(current_pos)
        self.last_position = current_pos

        if len(self.position_history) >= 10:
            recent_positions = list(self.position_history)[-10:]
            if recent_positions.count(current_pos) >= 3:
                self.emergency_mode = True
                self.emergency_cooldown = 15

        if self.emergency_cooldown > 0:
            self.emergency_cooldown -= 1
        if self.emergency_cooldown <= 0:
            self.emergency_mode = False

        nearest_ghost_dist = min(manhattan(current_pos, g) for g in state.ghosts) if state.ghosts else 999
        danger_replan = nearest_ghost_dist <= 4
        force_replan = self.stuck_counter > 3 or self.emergency_mode or danger_replan

        if self.replan_cooldown <= 0 or not self.path_cache or force_replan:
            if self.emergency_mode:
                self.mode = 'EVADE'
            else:
                self.mode = self.fuzzy.decide(state, self.init_pellets)

            if self.mode == 'EAT_POWER' and state.power:
                goals = set(state.power)
            elif self.mode == 'EVADE':
                all_walkable = {(x,y) for y,row in enumerate(LEVEL) for x,ch in enumerate(row) if ch != '#'}
                recent_positions = set(self.position_history)
                candidates = (all_walkable - recent_positions) or all_walkable
                dscore = []
                for c in candidates:
                    dmin = min(manhattan(c, g) for g in state.ghosts)
                    loop_penalty = -5 if c in recent_positions else 0
                    escape_routes = len(list(neighbors(*c)))
                    route_bonus = escape_routes * 2
                    total = dmin + loop_penalty + route_bonus
                    dscore.append((total, c))
                dscore.sort(reverse=True)
                goals = set(c for _, c in dscore[:30])
            else:
                recent_eaten = set(self.eaten_pellets_history)
                filtered = [p for p in state.pellets if p not in recent_eaten]
                goals = set(filtered) if filtered else set(state.pellets)

            walls_to_avoid = set(self.position_history) if self.emergency_mode else None
            path = self.pathfinder.astar(state.pac, goals, walls_block=walls_to_avoid, recent_path=self.recent_path)
            if not path and walls_to_avoid:
                path = self.pathfinder.astar(state.pac, goals, recent_path=self.recent_path)
            if not path:
                path = self.pathfinder.gbfs(state.pac, goals, recent_path=self.recent_path)
            self.path_cache = path[1:] if len(path) > 1 else []

            if self.emergency_mode or self.stuck_counter > 0:
                self.replan_cooldown = 1
            elif danger_replan:
                self.replan_cooldown = 2
            else:
                self.replan_cooldown = PACMAN_REPLAN_STEPS
        else:
            self.replan_cooldown -= 1

        if self.path_cache:
            nxt = self.path_cache.pop(0)
            return nxt, self.mode, list(self.path_cache)

        legal = list(neighbors(*state.pac))
        if legal:
            recent_positions = set(self.position_history)
            fresh_moves = [m for m in legal if m not in recent_positions]
            candidates = fresh_moves if fresh_moves else legal
            scored = []
            for m in candidates:
                ghost_dists = [manhattan(m, g) for g in state.ghosts]
                min_g = min(ghost_dists) if ghost_dists else 99
                score = min_g + random.uniform(0, 0.5)
                if state.pellets:
                    nearest_p = min(manhattan(m, p) for p in state.pellets)
                    score += (20 - nearest_p) * 0.1
                if state.power:
                    nearest_o = min(manhattan(m, p) for p in state.power)
                    score += (20 - nearest_o) * 0.2
                scored.append((score, m))
            scored.sort(reverse=True)
            return scored[0][1], self.mode, []
        return state.pac, self.mode, []



class Game:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        self.screen_width = info.current_w
        self.screen_height = info.current_h

        optimal_tile_width = self.screen_width // W
        optimal_tile_height = (self.screen_height - 200) // H
        self.tile_size = min(optimal_tile_width, optimal_tile_height, 48)

        self.game_width = W * self.tile_size
        self.game_height = H * self.tile_size
        self.hud_height = 150

        self.game_offset_x = (self.screen_width - self.game_width) // 2
        self.game_offset_y = (self.screen_height - self.game_height - self.hud_height) // 2

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Pacman vs Ghosts — AI vs AI (A*/GBFS + Minimax + Fuzzy Logic Hybrid)")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)


        s = INIT_STATE.copy()
        s._neighbors = neighbors
        self.state = s

        self.pac_agent = PacmanAgent(init_pellet_count=len(self.state.pellets))
        self.ghost_agents = [GhostAgent(0, GHOST_MINIMAX_DEPTH, neighbors),
                             GhostAgent(1, GHOST_MINIMAX_DEPTH, neighbors)]
        self.step_counter = 0
        self.score = 0
        self.overlay_path = []
        self.lives = 4
        self.ghost_behaviors = {}
        self.ghosts_eaten_in_power = 0
        self.last_power_time = 0
        self.last_death_time = 0
        self.deaths_since_power_spawn = 0
        self.initial_pellet_count = len(self.state.pellets)
        self.last_power_spawn_time = 0
        self.initial_power_count = len(self.state.power)

        self.animation_frame = 0
        self.pacman_direction = (1, 0)
        self.ghost_directions = [(0, 0), (0, 0)]

        self.invulnerable_timer = 0

    def update_logic(self):
        s = self.state
        self.animation_frame = (self.animation_frame + 1) % 10000

        pac_next, mode, overlay = self.pac_agent.choose_action(s)
        self.overlay_path = [s.pac] + overlay
        if pac_next != s.pac:
            self.pacman_direction = (pac_next[0] - s.pac[0], pac_next[1] - s.pac[1])

        if getattr(self.pac_agent, 'emergency_mode', False):
            mode += " [EMERGENCY]"
        elif self.pac_agent.stuck_counter > 0:
            mode += f" [STUCK:{self.pac_agent.stuck_counter}]"
        self.pac_agent.current_mode = mode
        s = self.apply_pac_move(s, pac_next)

        ghosts_next = []
        for i, ga in enumerate(self.ghost_agents):
            old_pos = s.ghosts[i]
            mv = ga.choose_action(s)
            if mv != old_pos:
                self.ghost_directions[i] = (mv[0] - old_pos[0], mv[1] - old_pos[1])
            self.ghost_behaviors[i] = ga.brain.current_behavior
            ghosts_next.append(mv)
            s = self.apply_ghost_move(s, mv, i)

        self.state = s
        self.step_counter += 1
        self.spawn_power_pellet_if_needed()

    def spawn_power_pellet_if_needed(self):
        pellets_remaining = len(self.state.pellets)
        pellets_eaten_ratio = 1.0 - (pellets_remaining / max(1, self.initial_pellet_count))
        if pellets_eaten_ratio < 0.1 or pellets_eaten_ratio > 0.85 or pellets_remaining <= 6:
            return

        should_spawn = False
        if len(self.state.power) == 0:
            if (self.step_counter - self.last_power_spawn_time) < 100:
                return
            if self.deaths_since_power_spawn >= 1:
                should_spawn = True
            elif self.last_death_time > 0:
                should_spawn = True
            elif self.lives <= 2:
                should_spawn = True
            elif self.last_power_time > 0 and (self.step_counter - self.last_power_time) > 150:
                should_spawn = True
            elif self.initial_power_count > 0 and self.step_counter > 200:
                should_spawn = True

        if not should_spawn:
            return

        empty_spots = []
        for y in range(1, H-1):
            for x in range(1, W-1):
                if LEVEL[y][x] != '#' and (x, y) not in self.state.power:
                    min_ghost_dist = min(manhattan((x, y), g) for g in self.state.ghosts)
                    if min_ghost_dist >= 3:
                        empty_spots.append((x, y))

        if len(empty_spots) < 5:
            for pellet_pos in self.state.pellets:
                min_ghost_dist = min(manhattan(pellet_pos, g) for g in self.state.ghosts)
                if min_ghost_dist >= 3:
                    empty_spots.append(pellet_pos)

        if len(empty_spots) < 3:
            for y in range(1, H-1):
                for x in range(1, W-1):
                    if LEVEL[y][x] != '#':
                        min_ghost_dist = min(manhattan((x, y), g) for g in self.state.ghosts)
                        if min_ghost_dist >= 2:
                            empty_spots.append((x, y))

        if empty_spots:
            spawn_pos = random.choice(empty_spots)
            new_power = set(self.state.power); new_power.add(spawn_pos)
            new_pellets = set(self.state.pellets); new_pellets.discard(spawn_pos)
            self.state = State(self.state.pac, self.state.ghosts, frozenset(new_pellets),
                               frozenset(new_power), self.state.scared_timer)
            self.deaths_since_power_spawn = 0
            self.last_power_spawn_time = self.step_counter

    def apply_pac_move(self, state: State, nxt):
        pac = nxt
        pellets = set(state.pellets)
        power = set(state.power)
        scared = state.scared_timer
        sc = 0

        if self.invulnerable_timer > 0:
            self.invulnerable_timer -= 1

        if pac in pellets:
            pellets.remove(pac)
            sc += 50
            self.pac_agent.eaten_pellets_history.append(pac)
        if pac in power:
            power.remove(pac)
            scared = 240
            sc += 500
            self.ghosts_eaten_in_power = 0
            self.last_power_time = self.step_counter

        ghosts = list(state.ghosts)
        if pac in ghosts and self.invulnerable_timer <= 0:
            if scared > 0:
                idx = ghosts.index(pac)
                ghosts[idx] = ghost_spawns[idx]
                sc += 750
                self.ghosts_eaten_in_power += 1
                sc += 1500
            else:
                sc -= 200
                pac = pac_spawn
                self.lives -= 1
                self.invulnerable_timer = 90
                self.pac_agent.path_cache = []
                self.last_death_time = self.step_counter
                self.deaths_since_power_spawn += 1

        self.score += sc
        new_state = State(pac, tuple(ghosts), frozenset(pellets), frozenset(power), scared)
        new_state._neighbors = neighbors  # keep hook for ghosts
        return new_state

    def apply_ghost_move(self, state: State, mv, ghost_idx):
        pac = state.pac
        ghosts = list(state.ghosts); ghosts[ghost_idx] = mv
        pellets = set(state.pellets)
        power = set(state.power)
        scared = max(0, state.scared_timer - 1) if ghost_idx == len(state.ghosts) - 1 else state.scared_timer
        sc = 0

        if mv == pac and self.invulnerable_timer <= 0:
            if scared > 0:
                ghosts[ghost_idx] = ghost_spawns[ghost_idx]
            else:
                pac = pac_spawn

        self.score += sc
        new_state = State(pac, tuple(ghosts), frozenset(pellets), frozenset(power), scared)
        new_state._neighbors = neighbors  # keep hook for ghosts
        return new_state

    def run(self):
        running = True
        game_over = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif game_over and event.key == pygame.K_SPACE:
                        running = False

            if not game_over:
                self.update_logic()
                self.renderer.draw_grid(self.state, self.overlay_path, self)
                self.renderer.draw_hud(self.step_counter, getattr(self.pac_agent, 'current_mode', self.pac_agent.mode),
                                       self.score, self.lives, self.ghost_behaviors, self)
                if (len(self.state.pellets) == 0 or self.step_counter >= MAX_GAME_STEPS or
                        self.lives <= 0 or self.score >= 30000):
                    game_over = True
            else:
                self.renderer.draw_grid(self.state, self.overlay_path, self)
                self.renderer.draw_hud(self.step_counter, getattr(self.pac_agent, 'current_mode', self.pac_agent.mode),
                                       self.score, self.lives, self.ghost_behaviors, self)
                self.renderer.draw_game_over(
                    won=((len(self.state.pellets) == 0 or self.score >= 30000) and self.lives > 0),
                    score=self.score,
                    steps=self.step_counter,
                    game_instance=self
                )
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()


if __name__ == '__main__':
    try:
        Game().run()
    except Exception as e:
        print("Error:", e)
        pygame.quit()
        sys.exit(1)
