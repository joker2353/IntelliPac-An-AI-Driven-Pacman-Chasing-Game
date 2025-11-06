import math
import random
from collections import deque

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



class _Pathfinder:
    def __init__(self, neighbors_func):
        self.neighbors = neighbors_func

    def astar(self, start, goals):
        if not goals:
            return []
        open_set = [(self._h(start, goals), 0, start, None)]
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
            for nx, ny in self.neighbors(*node):
                ng = gcost + 1
                if ng < g.get((nx, ny), 1e9):
                    g[(nx, ny)] = ng
                    nf = ng + self._h((nx, ny), goals)
                    open_set.append((nf, ng, (nx, ny), node))
        return []

    def _reconstruct(self, came, node):
        path = [node]
        while came[node] is not None:
            node = came[node]
            path.append(node)
        path.reverse()
        return path

    def _h(self, node, goals):
        if not goals:
            return 0
        return min(manhattan(node, g) for g in goals)



class GhostFuzzy:
    def __init__(self, ghost_index):
        self.ghost_index = ghost_index
        self.behavior_weights = {
            'AGGRESSIVE_PURSUIT': 0.0,
            'COORDINATED_HUNT': 0.0,
            'INTERCEPT': 0.0,
            'SCATTER': 0.0,
            'DEFENSIVE': 0.0
        }
        self.last_behavior = 'AGGRESSIVE_PURSUIT'
        self.behavior_momentum = 0

    # distance membership
    def mu_very_close(self, d):
        if d <= 1: return 1.0
        if d >= 4: return 0.0
        return (4 - d) / 3.0

    def mu_close(self, d):
        if d <= 2: return 0.0
        if d <= 4: return (d - 2) / 2.0
        if d <= 6: return (6 - d) / 2.0
        return 0.0

    def mu_medium(self, d):
        if d <= 5: return 0.0
        if d <= 7: return (d - 5) / 2.0
        if d <= 10: return (10 - d) / 3.0
        return 0.0

    def mu_far(self, d):
        if d <= 8: return 0.0
        if d <= 12: return (d - 8) / 4.0
        if d <= 15: return (15 - d) / 3.0
        return 0.0

    def mu_very_far(self, d):
        if d <= 12: return 0.0
        if d >= 20: return 1.0
        return (d - 12) / 8.0

  
    def mu_low_threat(self, t):
        if t <= 0.1: return 1.0
        if t >= 0.4: return 0.0
        return (0.4 - t) / 0.3

    def mu_medium_threat(self, t):
        if t <= 0.2: return 0.0
        if t <= 0.45: return (t - 0.2) / 0.25
        if t <= 0.7: return (0.7 - t) / 0.25
        return 0.0

    def mu_high_threat(self, t):
        if t <= 0.6: return 0.0
        if t >= 0.9: return 1.0
        return (t - 0.6) / 0.3

   
    def mu_good_coordination(self, c):
        if c <= 0.3: return 0.0
        if c <= 0.55: return (c - 0.3) / 0.25
        if c <= 0.8: return (0.8 - c) / 0.25
        return 0.0

    def mu_excellent_coordination(self, c):
        if c <= 0.6: return 0.0
        if c >= 0.9: return 1.0
        return (c - 0.6) / 0.3


    def mu_few_pellets(self, r):
        if r <= 0.1: return 1.0
        if r >= 0.4: return 0.0
        return (0.4 - r) / 0.3

    def mu_many_pellets(self, r):
        if r <= 0.6: return 0.0
        if r >= 0.9: return 1.0
        return (r - 0.6) / 0.3

    
    def calculate_pacman_threat(self, state):
        threat = 0.0
        if state.scared_timer > 0:
            threat += 0.6 * (state.scared_timer / 60.0)
        if state.power:
            min_power_dist = min(manhattan(state.pac, p) for p in state.power)
            if min_power_dist <= 3:
                threat += 0.4 * (1 - min_power_dist / 3.0)
        pellets_left = len(state.pellets)
        if pellets_left <= 10:
            threat += 0.3 * (1 - pellets_left / 10.0)
        return min(1.0, threat)

    def calculate_coordination(self, state):
        my_pos = state.ghosts[self.ghost_index]
        other_pos = state.ghosts[1 if self.ghost_index == 0 else 0]
        pac_pos = state.pac

        ghost_dist = manhattan(my_pos, other_pos)
        if 3 <= ghost_dist <= 6:
            dist_score = 1.0
        elif ghost_dist < 3:
            dist_score = ghost_dist / 3.0
        else:
            dist_score = max(0.0, 1.0 - (ghost_dist - 6) / 10.0)

        my_dist_to_pac = manhattan(my_pos, pac_pos)
        other_dist_to_pac = manhattan(other_pos, pac_pos)
        avg_dist = (my_dist_to_pac + other_dist_to_pac) / 2.0

        pincer_score = 0.0
        if avg_dist <= 6:
            pac_x, pac_y = pac_pos
            my_x, my_y = my_pos
            other_x, other_y = other_pos
            my_quad = (1 if my_x > pac_x else -1, 1 if my_y > pac_y else -1)
            other_quad = (1 if other_x > pac_x else -1, 1 if other_y > pac_y else -1)
            if my_quad != other_quad:
                pincer_score = 0.5

        return min(1.0, dist_score * 0.7 + pincer_score * 0.3)

    def calculate_inputs(self, state):
        pac_pos = state.pac
        my_pos = state.ghosts[self.ghost_index]
        other_ghost = state.ghosts[1 if self.ghost_index == 0 else 0]
        pacman_distance = manhattan(pac_pos, my_pos)
        other_ghost_distance = manhattan(my_pos, other_ghost)
        if state.power:
            power_pellet_distance = min(manhattan(pac_pos, p) for p in state.power)
        else:
            power_pellet_distance = 999
        pacman_power_level = self.calculate_pacman_threat(state)
        escape_routes = len(list(state._neighbors(*pac_pos))) if hasattr(state, "_neighbors") else 4
        pellets_remaining_ratio = len(state.pellets) / max(1, len(state.pellets) + 100)
        ghost_coordination = self.calculate_coordination(state)
        return {
            'pacman_distance': pacman_distance,
            'other_ghost_distance': other_ghost_distance,
            'power_pellet_distance': power_pellet_distance,
            'pacman_power_level': pacman_power_level,
            'escape_routes': escape_routes,
            'pellets_remaining_ratio': pellets_remaining_ratio,
            'ghost_coordination': ghost_coordination
        }

    def evaluate_rules(self, inputs):
        for k in self.behavior_weights:
            self.behavior_weights[k] = 0.0

        pac_dist = inputs['pacman_distance']
        ghost_dist = inputs['other_ghost_distance']
        threat = inputs['pacman_power_level']
        escape = inputs['escape_routes']
        pellets = inputs['pellets_remaining_ratio']
        coord = inputs['ghost_coordination']

        rule1 = min(self.mu_very_close(pac_dist), self.mu_low_threat(threat))
        self.behavior_weights['AGGRESSIVE_PURSUIT'] = max(self.behavior_weights['AGGRESSIVE_PURSUIT'], rule1 * 0.9)

        rule2 = min(self.mu_close(pac_dist), self.mu_good_coordination(coord))
        self.behavior_weights['AGGRESSIVE_PURSUIT'] = max(self.behavior_weights['AGGRESSIVE_PURSUIT'], rule2 * 0.8)

        rule3 = min(self.mu_medium(pac_dist), self.mu_excellent_coordination(coord))
        self.behavior_weights['COORDINATED_HUNT'] = max(self.behavior_weights['COORDINATED_HUNT'], rule3 * 0.9)

        rule4 = min(self.mu_close(pac_dist), 1.0 - min(1.0, max(0.0, escape / 4.0)))
        self.behavior_weights['COORDINATED_HUNT'] = max(self.behavior_weights['COORDINATED_HUNT'], rule4 * 0.7)

        rule5 = min(self.mu_few_pellets(pellets), self.mu_medium(pac_dist))
        self.behavior_weights['INTERCEPT'] = max(self.behavior_weights['INTERCEPT'], rule5 * 0.8)

        rule6 = min(self.mu_far(pac_dist), self.mu_good_coordination(coord))
        self.behavior_weights['INTERCEPT'] = max(self.behavior_weights['INTERCEPT'], rule6 * 0.6)

        rule7 = self.mu_high_threat(threat)
        self.behavior_weights['SCATTER'] = max(self.behavior_weights['SCATTER'], rule7 * 0.9)

        rule8 = 1.0 - self.mu_good_coordination(coord) if ghost_dist < 3 else 0.0
        self.behavior_weights['SCATTER'] = max(self.behavior_weights['SCATTER'], rule8 * 0.6)

        rule9 = min(self.mu_many_pellets(pellets), self.mu_very_far(pac_dist))
        self.behavior_weights['DEFENSIVE'] = max(self.behavior_weights['DEFENSIVE'], rule9 * 0.5)

        base_aggressive = 0.3
        self.behavior_weights['AGGRESSIVE_PURSUIT'] = max(self.behavior_weights['AGGRESSIVE_PURSUIT'], base_aggressive)

    def decide_behavior(self, state):
        inputs = self.calculate_inputs(state)
        self.evaluate_rules(inputs)
        if self.behavior_momentum > 0:
            self.behavior_momentum -= 1
            return self.last_behavior
        total = sum(self.behavior_weights.values())
        if total == 0:
            return 'AGGRESSIVE_PURSUIT'
        normalized = {b: w / total + random.uniform(0, 0.05) for b, w in self.behavior_weights.items()}
        chosen = max(normalized, key=normalized.get)
        if chosen != self.last_behavior:
            self.behavior_momentum = random.randint(3, 8)
        self.last_behavior = chosen
        return chosen



class GhostMinimax:
    def __init__(self, depth=1, ghost_index=0, neighbors_func=None):
        self.depth = max(1, depth)
        self.ghost_index = ghost_index
        self.neighbors = neighbors_func or (lambda x, y: [])
        self.last_positions = deque(maxlen=5)
        self.last_regions = deque(maxlen=5)
        self.last_direction = None
        self.stuck_counter = 0
        self.path_to_pacman = []
        self.path_recompute_cooldown = 0
        self.pathfinder = _Pathfinder(self.neighbors)

        self.fuzzy_brain = GhostFuzzy(ghost_index)
        self.current_behavior = 'AGGRESSIVE_PURSUIT'
        self.behavior_modifiers = {
            'AGGRESSIVE_PURSUIT': {'pursuit_weight': 1.0, 'coordination_weight': 0.02, 'risk_tolerance': 0.1},
            'COORDINATED_HUNT': {'pursuit_weight': 0.8, 'coordination_weight': 0.05, 'risk_tolerance': 0.08},
            'INTERCEPT': {'pursuit_weight': 0.7, 'coordination_weight': 0.04, 'risk_tolerance': 0.07},
            'SCATTER': {'pursuit_weight': 0.4, 'coordination_weight': -0.02, 'risk_tolerance': 0.04},
            'DEFENSIVE': {'pursuit_weight': 0.5, 'coordination_weight': 0.03, 'risk_tolerance': 0.05}
        }

  
    def legal_moves_from(self, pos):
        moves = [(nx, ny) for nx, ny in self.neighbors(*pos)]
        if self.last_positions:
            moves.sort(key=lambda m: m not in list(self.last_positions)[-3:])
        return moves

    def get_region(self, pos):
        x, y = pos
        return (x // 3, y // 3)

    def get_direction(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if abs(dx) > abs(dy):
            return (1 if dx > 0 else -1, 0)
        return (0, 1 if dy > 0 else -1)

    def update_movement_history(self, pos):
        self.last_positions.append(pos)
        if len(self.last_positions) >= 4:
            if self.last_positions[-1] == self.last_positions[-3] and \
               self.last_positions[-2] == self.last_positions[-4]:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0

 
    def eval_state(self, state, me_idx):
        pac = state.pac
        my_pos = state.ghosts[me_idx]
        other_ghost = state.ghosts[1 if me_idx == 0 else 0]

        self.current_behavior = self.fuzzy_brain.decide_behavior(state)
        modifiers = self.behavior_modifiers[self.current_behavior]

        current_region = self.get_region(my_pos)
        region_penalty = 0
        if self.last_regions:
            recent = list(self.last_regions)[-4:]
            if current_region in recent:
                count = recent.count(current_region)
                region_penalty = 40 * (2 ** count)

        stuck_penalty = 30 * self.stuck_counter if self.stuck_counter > 0 else 0

        direction_penalty = 0
        if len(self.last_positions) >= 2:
            current_dir = self.get_direction(self.last_positions[-1], my_pos)
            if self.last_direction and current_dir != self.last_direction:
                direction_penalty = 20

        my_dist = manhattan(pac, my_pos)
        other_dist = manhattan(pac, other_ghost)
        ghost_sep = manhattan(my_pos, other_ghost)
        clustering_penalty = 20 if ghost_sep <= 1 else 0
        coordination_bonus = 10 if 3 <= ghost_sep <= 6 else 0

        pursuit_weight = modifiers['pursuit_weight']
        pursuit_score = (20 - my_dist) * pursuit_weight

        risk_tolerance = modifiers['risk_tolerance']
        if my_dist <= 3 and risk_tolerance > 0.5:
            pursuit_score *= (1 + risk_tolerance)

        intercept_score = 0
        if state.pellets:
            nearest_food_to_pac = min((manhattan(pac, p) for p in state.pellets))
            if nearest_food_to_pac < 3:
                target_food = min(state.pellets, key=lambda p: manhattan(pac, p))
                my_dist_to_food = manhattan(my_pos, target_food)
                if my_dist_to_food < nearest_food_to_pac:
                    intercept_score = 30

        scared = state.scared_timer > 0
        if scared:
            if self.current_behavior == 'SCATTER':
                score = my_dist * 3
            elif self.current_behavior == 'DEFENSIVE':
                score = my_dist * 1.5
                score -= clustering_penalty
            else:
                score = my_dist * 2
            score -= region_penalty + stuck_penalty + direction_penalty * 0.5
        else:
            if self.current_behavior == 'AGGRESSIVE_PURSUIT':
                score = pursuit_score
                if my_dist <= 2:
                    score += 5 * risk_tolerance
            elif self.current_behavior == 'COORDINATED_HUNT':
                score = pursuit_score * 0.8
                score += coordination_bonus * modifiers['coordination_weight'] * 40
                score -= clustering_penalty
                if my_dist <= 4 and other_dist <= 4 and 2 <= ghost_sep:
                    score += 3
            elif self.current_behavior == 'INTERCEPT':
                score = pursuit_score * 0.6
                if intercept_score > 0:
                    score += intercept_score * 2.5
                escape_routes = len(list(state._neighbors(*pac))) if hasattr(state, "_neighbors") else 4
                if escape_routes <= 2:
                    score += 3
            elif self.current_behavior == 'SCATTER':
                score = ghost_sep * 2 + my_dist * 0.5 - region_penalty * 2
            elif self.current_behavior == 'DEFENSIVE':
                score = pursuit_score * 0.4
                score += coordination_bonus * modifiers['coordination_weight'] * 30
                score -= clustering_penalty
                if state.power:
                    nearest_power = min(manhattan(my_pos, p) for p in state.power)
                    if nearest_power <= 3:
                        score += 40

            score -= region_penalty + stuck_penalty + direction_penalty
            if self.stuck_counter > 0:
                stuck_boost = (20 - my_dist) * (self.stuck_counter + 1) * risk_tolerance
                score += stuck_boost

        if pac in state.ghosts and not scared:
            return 999999
        if len(state.pellets) == 0:
            return -999999
        return score

    def next_state_after(self, state, pac_next=None, ghost_next=None, ghost_idx=0):
        pac = state.pac if pac_next is None else pac_next
        ghosts = list(state.ghosts)
        if ghost_next is not None:
            ghosts[ghost_idx] = ghost_next
        pellets = set(state.pellets)
        power = set(state.power)
        scared_timer = state.scared_timer
        if pac in pellets:
            pellets.remove(pac)
        if pac in power:
            power.remove(pac)
            scared_timer = 240
        # copy but preserve the neighbors function reference if present
        new_state = type("GhostEvalState", (), {})()
        new_state.pac = pac
        new_state.ghosts = tuple(ghosts)
        new_state.pellets = frozenset(pellets)
        new_state.power = frozenset(power)
        new_state.scared_timer = scared_timer
        if hasattr(state, "_neighbors"):
            new_state._neighbors = state._neighbors
        return new_state

    def choose(self, state, my_index):
        try:
            current_pos = state.ghosts[my_index]
            if current_pos and all(isinstance(x, (int, float)) for x in current_pos):
                self.update_movement_history(current_pos)
                current_region = self.get_region(current_pos)
                self.last_regions.append(current_region)
                if len(self.last_regions) > 5:
                    self.last_regions.popleft()
        except Exception:
            self.last_positions.clear()
            self.last_regions.clear()
            self.stuck_counter = 0

        current_depth = max(1, self.depth + random.choice([-1, -1, 0, 0, 1]))

        def max_value(s, depth, alpha, beta):
            if depth == 0:
                return self.eval_state(s, my_index), None
            best_score = -1e9
            best_move = None
            moves = self.legal_moves_from(s.ghosts[my_index])
            moves = list(moves)
            random.shuffle(moves)
            for mv in moves:
                ns = self.next_state_after(s, ghost_next=mv, ghost_idx=my_index)
                sc, _ = min_value(ns, depth - 1, alpha, beta)
                if -9999 < sc < 9999:
                    sc += random.uniform(-5, 5)
                if sc > best_score:
                    best_score, best_move = sc, mv
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score, best_move

        def min_value(s, depth, alpha, beta):
            best_score = 1e9
            best_move = None
            moves = self.legal_moves_from(s.pac)
            moves = list(moves)
            random.shuffle(moves)
            for mv in moves:
                ns = self.next_state_after(s, pac_next=mv)
                sc, _ = max_value(ns, depth, alpha, beta)
                if sc < best_score:
                    best_score, best_move = sc, mv
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score, best_move

        _, move = max_value(state, current_depth, -1e9, 1e9)

        if self.path_recompute_cooldown <= 0 or not self.path_to_pacman:
            path = self.pathfinder.astar(state.ghosts[my_index], {state.pac})
            if path and len(path) > 1:
                self.path_to_pacman = path[1:]
            else:
                self.path_to_pacman = []
            self.path_recompute_cooldown = 5
        else:
            self.path_recompute_cooldown -= 1

        if self.path_to_pacman and self.stuck_counter < 2:
            path_move = self.path_to_pacman[0]
            if path_move in self.legal_moves_from(state.ghosts[my_index]):
                move = path_move
                self.path_to_pacman.pop(0)
            else:
                self.path_to_pacman = []
                self.path_recompute_cooldown = 0

        if move and random.random() < 0.35:
            lm = self.legal_moves_from(state.ghosts[my_index])
            if lm:
                move = random.choice(lm)

        if move is None or self.stuck_counter >= 2:
            lm = self.legal_moves_from(state.ghosts[my_index])
            if lm:
                if random.random() < 0.40:
                    move = random.choice(lm)
                else:
                    distances = [(manhattan(mv, state.pac), mv) for mv in lm]
                    min_dist = min(d for d, _ in distances)
                    closest = [mv for d, mv in distances if d == min_dist]
                    move = random.choice(closest) if closest else state.ghosts[my_index]
            else:
                move = state.ghosts[my_index]

        if move and state.ghosts[my_index] != move:
            self.last_direction = self.get_direction(state.ghosts[my_index], move)

        return move


class GhostAgent:

    def __init__(self, index, depth, neighbors_func):
        self.index = index
        self.brain = GhostMinimax(depth=depth, ghost_index=index, neighbors_func=neighbors_func)

    def choose_action(self, state):
        return self.brain.choose(state, self.index)
