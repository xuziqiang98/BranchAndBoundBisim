import torch
from SelectTree import CustomNodeSelector
from pyscipopt import Model
import numpy as np
import pyscipopt as scip
from scipy.spatial.distance import cdist
from utils import powernorm
from tsp_mutator import do_mutation
import tsplib95 as tsplib
from scipy.io import mmread

def generate_test_data(seed):
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    model = Model()  # model name is optional
    ls = []
    for i in range(300):
        x = model.addVar(f"x{i}", vtype="INTEGER")
        y = model.addVar(f"y{i}", vtype="INTEGER")
        model.addCons(2*x - y + sum([torch.randint(-50,50,(1,),generator=g_cpu).item()*j for j in ls[-20:-4]]) >= 0)
        if torch.rand(1,generator=g_cpu)<0.5:
            model.addCons(-1<= (x<= 1))
            ls.append(x)
        else:
            model.addCons(-5<= (y<= 500))
            ls.append(y)
    #model.addCons(sum([torch.randn(1).item()*i for i in ls])**2 >= 0.5)
    model.setObjective(sum([torch.randn(1,generator=g_cpu).item()*i for i in ls]))
    model.writeProblem(f"model-{seed}.cip")
    return f"model-{seed}.cip"


class Wrapper:
    def __init__(self, tspobj):
        self.tspobj = tspobj
        self.nodes = list(tspobj.get_nodes())
    def __getitem__(self, key):
        if isinstance(key,tuple):

            key = self.nodes[key[0]], self.nodes[key[1]]
            i= self.tspobj.get_weight(*key)
            return i
        elif isinstance(key, int):
            return [self.tspobj.get_weight(key,i) for i in range(self.tspobj.dimension)]
    @property
    def shape(self):
        return (self.tspobj.dimension, self.tspobj.dimension)


def make_tsplib(location:str):
# def make_tsplib():
    # location = "tsplib/"
    problem = tsplib.load(location)
    size = problem.dimension
    ds= Wrapper(problem)#np.array([problem.get_weight(*i) for i in problem.get_edges()]).reshape(size,size)
    if size >= 1_000:
        return None
    return make_tsp(distances=ds)

def make_tsp(seed=None, distances=None):
    """
    USE MTZ formulation
    """
    #g_cpu = torch.Generator()
    #if seed is not None:
    #    g_cpu.manual_seed(seed)
    # Define a distance matrix for the cities
    dist_matrix = distances
    if distances is None:
        size = np.random.randint(75,90)
        d = do_mutation(powernorm(torch.randn(size,2,)*size,0.5).numpy())
        y = np.random.rand(size,size)
        random_offset = y-np.diag(y)*np.eye(len(y))
        dist_matrix = cdist(d,d)#+random_offset*2
    #print("TSP size",size)
    # Create a SCIP model
    model = Model("TSP")

    # Define variables
    num_cities = dist_matrix.shape[0]
    x = {}

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                x[i,j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
    u={}
    for i in range(1,num_cities):
        u[i] = model.addVar(vtype="I", name=f"u_{i}")
        model.addCons(1<=(u[i]<= num_cities-1), name=f"u_{i}_constraint")

    # Define constraints
    # Each city must be visited exactly once
    for i in range(num_cities):
        model.addCons(scip.quicksum(x[i,j] for j in range(num_cities) if j != i) == 1, name=f"city_{i}_visited_origin")
    for j in range(num_cities):
        model.addCons(scip.quicksum(x[i,j] for i in range(num_cities) if j != i) == 1, name=f"city_{j}_visited_dest")
    # There should be no subtours
    for i in range(1,num_cities):
        for j in range(1,num_cities):
            if i != j:
                model.addCons(u[i] - u[j] + (num_cities - 1)*x[i,j]<= num_cities-2, name=f"no_subtour_{i}_{j}")
    

    # Define objective
    model.setObjective(scip.quicksum(dist_matrix[i,j] * x[i,j] for i in range(num_cities) for j in range(num_cities) if i!=j), sense="minimize")
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    return model

def create_knapsack_instance(seed=None):
    """
    Creates a knapsack instance with the given values, weights, and capacity.
    Returns a PySCIPOpt model object.
    """
    sz = 15_000
    values = torch.randint(1,1000,(sz,)).numpy()
    weights = torch.randint(1,1000,(sz,)).numpy()
    capacity = weights.sum()/10
    # Create a SCIP model
    model = scip.Model("Knapsack")

    # Define variables
    n = len(values)
    x = {}

    for i in range(n):
        x[i] = model.addVar(vtype="B", name=f"x_{i}")

    # Define constraints
    # The total weight cannot exceed the capacity
    model.addCons(scip.quicksum(weights[i] * x[i] for i in range(n)) <= capacity, name="capacity_constraint")

    # Define objective
    model.setObjective(scip.quicksum(values[i] * x[i] for i in range(n)), sense="maximize")
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    #print("Knapsack")
    # Return the model
    return model

def cutting_stock(seed=None):
    # Define problem parameters
    patterns = 5_000
    orders = 100
    orders_sizes = 100+2*torch.rand(orders,).numpy()
    item_cost = 10+torch.rand(patterns,).numpy()
    item_lengths = 10+2*torch.rand(patterns,orders).numpy()

    # Create PySCIPOpt model
    model = Model("cutting_stock")
    x=[]
    for i in range(patterns):
        x.append(model.addVar(vtype="I", name=f"x_{i}"))
    for j in range(orders):
        model.addCons(scip.quicksum(x[i]*item_lengths[i,j] for i in range(patterns)) >= orders_sizes[j])
        
    for i in range(patterns):
        model.addCons(x[i]>=0)

    # Define the objective function to minimize the total number of stock pieces used
    model.setObjective(scip.quicksum(x[i]*item_cost[i] for i in range(patterns)), sense="minimize")
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    return model

def subset_sum(seed=None):
    print("SUBSET MADE")
    n = 4000
    model = scip.Model("subset-sum")
    c = torch.randint(500,2000,(1,)).numpy()
    w = torch.randn(n).numpy()*2000
    x={}
    for i in range(n):
        x[i] = model.addVar(vtype="B", name=f"x_{i}")
    model.addCons(scip.quicksum(w[i] * x[i] for i in range(n)) <= c, name="capacity_constraint")
    model.setObjective(scip.quicksum(w[i] * x[i] for i in range(n)), sense="maximize")
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    # Return the model
    return model


def read_lp(location: str):
    model = Model()
    model.readProblem(location)
    return model

def generate_production_planning_instance(seed=None):
    # Create PySCIPOpt model
    model = Model("production_planning")
    num_products =64
    num_resources=256
    max_demand=80
    max_capacity=90
    weightings = np.ones(num_products,)
    # Create decision variables
    x = {}
    for i in range(num_products):
        for j in range(num_resources):
            x[(i, j)] = model.addVar(vtype="I", name=f"x_{i}_{j}")
            model.addCons(x[(i, j)]>=0)

    # Define demand constraints
    for i in range(num_products):
        model.addCons(scip.quicksum(x[(i, j)] for j in range(num_resources)) >= np.random.randint(1, max_demand))

    # Define capacity constraints
    for j in range(num_resources):
        model.addCons(scip.quicksum(x[(i, j)] for i in range(num_products)) <= np.random.randint(1, max_capacity))

    # Define the objective function
    model.setObjective(scip.quicksum(x[(i, j)]*w for i,w in enumerate(weightings) for j in range(num_resources)), sense="maximize")
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")

    return model


def capacitated_facility_location(seed=None):
    """
    Solves a capacitated facility location problem using PySCIPOpt.
    Inputs:
        - num_facilities: number of potential facility locations
        - num_customers: number of demand points
        - facility_costs: list of length num_facilities containing the fixed costs of opening each facility
        - customer_demands: list of length num_customers containing the demand of each customer
        - facility_capacities: list of length num_facilities containing the capacity of each facility
        - customer_facility_costs: list of lists of length num_facilities, where customer_facility_costs[i][j] is the cost of satisfying the demand of customer i from facility j
    """
    print("locations")
    num_facilities  = int(100)
    num_customers  = int(100)
    facility_costs = 3000* torch.ones((num_facilities,)).numpy()
    customer_demands = torch.ones((num_customers,)).numpy()
    facility_capacities = 10_000_000*torch.ones((num_facilities,)).numpy()
    customer_facility_costs = torch.ones((num_customers, num_facilities)).numpy()*3000
    for c in range(num_customers):
        customer_facility_costs[c, np.random.choice(num_facilities,size=10)] = np.random.randint(0,4,size=10)
    # Create a SCIP model
    model = scip.Model("Capacitated Facility Location")

    # Define variables
    facility_open = {}
    for i in range(num_facilities):
        facility_open[i] = model.addVar(vtype="B", name=f"f_{i}")
    customer_assign = {}
    for i in range(num_customers):
        customer_assign[i] = {}
        for j in range(num_facilities):
            customer_assign[i][j] = model.addVar(vtype="C", name=f"a_{i}_{j}")

    # Define constraints
    # Demand satisfaction constraints
    for i in range(num_customers):
        model.addCons(scip.quicksum(customer_assign[i][j] for j in range(num_facilities)) == customer_demands[i], name=f"demand_constraint_{i}")
    # Capacity constraints
    for j in range(num_facilities):
        model.addCons(scip.quicksum(customer_assign[i][j] for i in range(num_customers)) <= facility_capacities[j] * facility_open[j], name=f"capacity_constraint_{j}")

    # Define objective
    model.setObjective(
        scip.quicksum(facility_costs[j] * facility_open[j] for j in range(num_facilities)) +
        scip.quicksum(customer_facility_costs[i][j] * customer_assign[i][j] for i in range(num_customers) for j in range(num_facilities)),
        sense="minimize"
    )

    # Solve the model
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    return model

def _make_dummy_model(seed=None):
    model = Model()  # model name is optional
    ls = []
    for i in range(300):
        x = model.addVar(f"x{i}", vtype="INTEGER")
        y = model.addVar(f"y{i}", vtype="INTEGER")
        model.addCons(2*x - y + sum([torch.randint(-50,50,(1,)).item()*j for j in ls[-30:-4]]) >= 0)
        if torch.rand(1)<0.5:
            model.addCons(-1<= (x<= 1))
            ls.append(x)
        else:
            model.addCons(-5<= (y<= 500))
            ls.append(y)
    #model.addCons(sum([torch.randn(1).item()*i for i in ls])**2 >= 0.5)
    model.setObjective(sum([torch.randn(1).item()*i for i in ls]))
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    #model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
    #model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
    # model.disablePropagation()
    return model

def make_mtx(location:str):
    # with open(location, 'r') as f:
    #     lines = f.readlines()
    #     # 跳过第一行（以%%开头的那行）
    #     data_lines = lines[1:]
    #     # 从第二行获取矩阵的行数和列数（假设格式相对简单，第二行是类似'3 3'表示3行3列这样的格式）
    #     n_rows, n_cols, _ = map(int, data_lines[0].split())
    #     matrix = np.zeros((n_rows, n_cols), dtype=int)
    #     for line in data_lines[1:]:
    #         parts = line.split()
    #         if len(parts) >= 2:
    #             u, v = map(int, parts[:2])
    #             # 添加索引范围验证，避免越界
    #             if 1 <= u <= n_rows and 1 <= v <= n_cols:
    #                 matrix[u - 1][v - 1] = 1
    #                 if n_rows == n_cols:  # 处理无向图情况，保证对称
    #                     matrix[v - 1][u - 1] = 1
    # 读取 .mtx 文件为稀疏矩阵
    sparse_matrix = mmread(location)
    
    # 确保是无向图（矩阵对称）
    adjacency_matrix = sparse_matrix + sparse_matrix.T
    
    # 转换为 NumPy 的邻接矩阵格式
    adjacency_matrix = adjacency_matrix.toarray()
    
    # 将非零元素设置为 1（表示无权图）
    adjacency_matrix[adjacency_matrix != 0] = 1.

    return make_dsp(matrix=adjacency_matrix)

def make_dsp(seed=None, matrix=None):
    # Create a SCIP model
    model = Model("DSP")

    if matrix is None:
        n = np.random.randint(800, 1000)
        get_w = lambda: np.random.choice([0, 1])
        density = np.random.uniform()
        matrix = np.zeros((n, n))
        
        # Generate the graph
        for i in range(n):
            for j in range(i):
                if np.random.uniform() < density:
                    w = get_w()
                    matrix[i, j] = w
                    matrix[j, i] = w
    else:
        n = matrix.shape[0]
    
    # Define variables
    x = {}
    for i in range(n):
        x[i] = model.addVar(vtype="B", name=f"x_{i}")
    
    # Define constraints
    for v in range(n):
        if np.sum(matrix[v]) == 0:  # Check for isolated nodes
            model.addCons(x[v] == 1, f"isolated_{v}")
        else:
            neighbors = [u for u in range(n) if matrix[v][u] == 1]
            model.addCons(x[v] + scip.quicksum(x[u] for u in neighbors) >= 1, f"dominate_{v}")
    
    # Define objective
    model.setObjective(scip.quicksum(x[v] for v in range(n)), sense="minimize")
    
    # If a seed is provided, write the problem to a file
    if seed is not None:
        model.writeProblem(f"model-{seed}.cip")
    
    return model
