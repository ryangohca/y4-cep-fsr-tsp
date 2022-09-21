import random
import math
import itertools
from functools import lru_cache
from typing import List, Tuple, TypeAlias, Callable

PointList: TypeAlias = "List[Tuple[int, int]]"

# Helper data structure for the shortest_edge algortihm
class DisjointSetUnion:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.setSize = [1] * n
    
    def find_parent(self, node: int) -> int:
        if self.parent[node] == node:
            return node
        self.parent[node] = self.find_parent(self.parent[node])
        return self.parent[node]

    def merge_sets(self, node_x: int, node_y: int) -> None:
        parent_x = self.find_parent(node_x)
        parent_y = self.find_parent(node_y)
        if parent_x == parent_y:
            return
        if self.setSize[parent_x] < self.setSize[parent_y]:
            parent_x, parent_y = parent_y, parent_x
        self.parent[parent_y] = parent_x
        self.setSize[parent_x] += self.setSize[parent_y]

    def is_same_set(self, node_x: int, node_y: int) -> bool:
        return self.find_parent(node_x) == self.find_parent(node_y)

    def find_size(self, node: int) -> int:
        return self.setSize[self.find_parent(node)]

def generate_points(n: int, maxx: int = 300, maxy: int=300) -> PointList:
    return [(random.randint(0, maxx), random.randint(0, maxy)) for i in range(n)]

# max n = 9, O(N*N!)
def brute_force(points : PointList) -> Tuple[float, PointList]:
    min_dist = 1e18
    min_travel_path = []
    for path in itertools.permutations(points):
        curr_dist = math.dist(path[-1], path[0])
        for i in range(len(path)-1):
            curr_dist += math.dist(path[i], path[i+1])
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_travel_path = list(path)
    return min_dist, min_travel_path

brute_force.name = "Brute Force"

# max n = 16, O(N^2*2^N)
def dynamic_programming(points: PointList) -> Tuple[float, PointList]:
    n = len(points)
    @lru_cache(None)
    def dp(idx: int, bitmask: int) -> float: #optimised to support higher n
        if bitmask == (1<<n)-1:
            return math.dist(points[idx], points[0])
        min_dist = 1e18
        for i in range(n):
            if bitmask & (1<<i): 
                continue
            min_dist = min(min_dist, dp(i, bitmask | (1<<i)) + math.dist(points[idx], points[i]))
        return min_dist

    min_dist = dp(0, 0)

    def reconstruct_path() -> PointList:
        curr_path = [0]
        curr_bitmask = (0 | (1<<0))
        while len(curr_path) != n:
            curr_point_idx = curr_path[-1]
            next_point = 0
            curr_min_dist = 1e18
            for i in range(n):
                if curr_bitmask & (1<<i):
                    continue
                this_dist = dp(i, curr_bitmask | (1<<i)) + math.dist(points[curr_point_idx], points[i])
                if curr_min_dist > this_dist:
                    next_point = i
                    curr_min_dist = this_dist
            curr_path.append(next_point)
            curr_bitmask |= (1<<next_point)
        return [points[i] for i in curr_path]

    min_travel_path = reconstruct_path()
    return min_dist, min_travel_path

dynamic_programming.name = "Dynamic Programming"

# O(N^3)
# O(N^2) if testAll is False, ie. we test with only 1 point as starting point.
def nearest_neighbours(points: PointList, testAll: bool=True) -> Tuple[float, PointList]:
    n = len(points)
    dist_matrix = [[0.0] * n for i in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = math.dist(points[i], points[j])
    
    min_dist = 1e18
    min_travel_path = []
    for start_point in range(n):
        this_dist = 0
        this_path = [start_point]
        visited_nodes = {start_point}
        for i in range(n-1):
            nearest_point = -1
            for next_point in range(n):
                if next_point in visited_nodes:
                    continue
                if nearest_point == -1 or dist_matrix[this_path[-1]][nearest_point] > dist_matrix[this_path[-1]][next_point]:
                    nearest_point = next_point
            this_dist += dist_matrix[this_path[-1]][nearest_point]
            this_path.append(nearest_point)
            visited_nodes.add(nearest_point)
        this_dist += dist_matrix[this_path[0]][this_path[-1]]
        if min_dist > this_dist:
            min_dist = this_dist
            min_travel_path = this_path
        if not testAll:
            break
    return min_dist, [points[i] for i in min_travel_path]

nearest_neighbours.name = "Nearest Neighbours"
nearest_neighbours_test_once = lambda points: nearest_neighbours(points, testAll=False)
nearest_neighbours_test_once.name = "Nearest Neighbours (Test 1 node as starting point only)"

# O(N^2logN)
def shortest_edge(points: PointList) -> Tuple[float, PointList]:
    n = len(points)
    dsu = DisjointSetUnion(n)
    degree = [0] * n
    
    def isValidJoin(node_x, node_y):
        if degree[node_x] >= 2 or degree[node_y] >= 2:
            return False
        if dsu.is_same_set(node_x, node_y):
            return dsu.find_size(node_x) == n
        return True

    def join(node_x, node_y):
        dsu.merge_sets(node_x, node_y)
        degree[node_x] += 1
        degree[node_y] += 1
    
    edges = [(math.dist(points[i], points[j]), i, j) for i in range(n) for j in range(i)]
    edges.sort()

    selected_edges = []
    min_cost = 0.0
    for dist, node_x, node_y in edges:
        if isValidJoin(node_x, node_y):
            selected_edges.append((node_x, node_y))
            min_cost += dist
            join(node_x, node_y)
        if len(selected_edges) == n:
            break
    
    # find valid construction
    min_travel_path = []
    adjlist = [[] for i in range(n)]
    visited = [False] * n
    for node_x, node_y in selected_edges:
        adjlist[node_x].append(node_y)
        adjlist[node_y].append(node_x)
    
    def dfs(x):
        if visited[x]:
            return
        min_travel_path.append(points[x])
        visited[x] = True
        for nxt in adjlist[x]:
            dfs(nxt)
    
    dfs(0)
    
    return min_cost, min_travel_path

shortest_edge.name = "Shortest Edge"

# O(cN + N^2), c = number of swaps made
# c is not bounded in polynomial time in the worse case scenario, but small enough for the average case.
def two_opt(points: PointList, generate_path_func: Callable) -> Tuple[float, PointList]:
    n = len(points)
    min_dist, curr_travel_path = generate_path_func(points)
    while True:
        swap = False
        for i in range(n):
            # consider removing edge connecting curr_travel_path[i-1] and curr_travel_path[i] (include wraparound)
            opt_j = -1
            for j in range(i+2, n):
                # consider removing edge connecting curr_travel_path[j-1] and curr_travel_path[j]
                if math.dist(curr_travel_path[i-1], curr_travel_path[i]) + math.dist(curr_travel_path[j-1], curr_travel_path[j]) > math.dist(curr_travel_path[i-1], curr_travel_path[j-1]) + math.dist(curr_travel_path[i], curr_travel_path[j]):
                    opt_j = j
                    break
            if opt_j != -1:
                min_dist = min_dist - (math.dist(curr_travel_path[i-1], curr_travel_path[i]) + math.dist(curr_travel_path[opt_j-1], curr_travel_path[opt_j])) + (math.dist(curr_travel_path[i-1], curr_travel_path[opt_j-1]) + math.dist(curr_travel_path[i], curr_travel_path[opt_j]))
                curr_travel_path[i:opt_j] = reversed(curr_travel_path[i:opt_j])
                swap = True
                break
        if not swap:
            break
    return min_dist, curr_travel_path

def check_ans(path, ans):
    curr_dist = 0
    for i in range(len(path)):
        curr_dist += math.dist(path[i], path[i-1])
    return math.isclose(curr_dist, ans)

two_opt_nearest_neighbours = lambda points: two_opt(points, nearest_neighbours)
two_opt_nearest_neighbours.name = "2-opt (Nearest Neighbours)"
two_opt_nearest_neighbours_test_once = lambda points: two_opt(points, nearest_neighbours_test_once)
two_opt_nearest_neighbours_test_once.name = "2-opt (Nearest Neighbours with testing of only 1 starting point)"
two_opt_shortest_edge = lambda points: two_opt(points, shortest_edge)
two_opt_shortest_edge.name = "2-opt (Shortest Edge)"

def test_functions(n: int, funcs: List[Callable], reduced_output: bool = True, maxx: int = 20, maxy: int = 20):
    test_points = generate_points(n, maxx=maxx, maxy=maxy)
    if reduced_output:
        print(f"Test points are generated with n={n}, coordinates are within (0, 0) and ({maxx}, {maxy})")
    else:
        print(f"Test points are: {test_points}, n={n}, coordinates are within (0, 0) and ({maxx}, {maxy})")
    print()
    for func in funcs:
        print(func.name + ":")
        min_dist, travel_path = func(test_points)
        assert check_ans(travel_path, min_dist)
        if reduced_output:
            print(f"Minimum distance is {min_dist:.5f}")
        else:
            print(f"Minimum distance is {min_dist:.5f}, with path {travel_path}")
        print()

def test():
    random.seed(17042003)
    test_functions(9, [brute_force, dynamic_programming])
    test_functions(15, [dynamic_programming, 
                        nearest_neighbours_test_once, 
                        nearest_neighbours, 
                        shortest_edge, 
                        two_opt_nearest_neighbours_test_once, 
                        two_opt_nearest_neighbours, 
                        two_opt_shortest_edge])
    #test_functions(400, [nearest_neighbours_test_once, nearest_neighbours, shortest_edge, two_opt_nearest_neighbours_test_once, two_opt_nearest_neighbours, two_opt_shortest_edge], reduced_output=True, maxx=100, maxy=100)

if __name__ == "__main__":
    test()
