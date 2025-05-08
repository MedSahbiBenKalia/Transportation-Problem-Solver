from gurobipy import Model, GRB, LinExpr
import numpy as np

class TransportModel:
    def __init__(self):
        self.prob = None
        self.x = None
        self.supply_constraints = None
        self.demand_constraints = None
        self.n_origins = 0
        self.n_destinations = 0
        self.costs = None
        self.status = None
        self.status_message = None


    def build_model(self, costs, supplies, demands):
        """
        Builds the transportation model.

        Args:
            costs (2D array): Cost matrix where costs[i][j] is the cost of transporting from origin i to destination j.
            supplies (1D array): Supply capacities for each origin.
            demands (1D array): Demand requirements for each destination.

        Returns:
            list: A list of warning messages about adjustments made to the inputs.
        """
        messages = []

        # Validate inputs and adjust if necessary
        if len(costs) != len(supplies):
            messages.append("Warning: Number of rows in costs does not match the length of supplies. Adjusting...")
            min_len = min(len(costs), len(supplies))
            costs = costs[:min_len]
            supplies = supplies[:min_len]

        if len(costs[0]) != len(demands):
            messages.append("Warning: Number of columns in costs does not match the length of demands. Adjusting...")
            min_len = min(len(costs[0]), len(demands))
            costs = [row[:min_len] for row in costs]
            demands = demands[:min_len]


        if sum(supplies) < sum(demands):
            messages.append("Warning: Total supply is less than total demand. Adjusting supply to match demand...")
        

        self.n_origins = len(supplies)
        self.n_destinations = len(demands)
        self.costs = costs

        # Initialize Gurobi model
        self.prob = Model("TransportProblem")
        self.prob.setParam('OutputFlag', 0)  # Suppress Gurobi output for cleaner interface

        # Decision variables
        self.x = {}
        for i in range(self.n_origins):
            for j in range(self.n_destinations):
                self.x[(i, j)] = self.prob.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}_{j}")

        # Objective: Minimize total cost
        objective = LinExpr()
        for i in range(self.n_origins):
            for j in range(self.n_destinations):
                objective.add(self.x[(i, j)], self.costs[i][j])
        self.prob.setObjective(objective, GRB.MINIMIZE)

        # Constraints
        self.supply_constraints = {}
        self.demand_constraints = {}

        total_supply = sum(supplies)
        total_demand = sum(demands)

        if total_supply >= total_demand:
            # Demand constraints: == demand
            for j in range(self.n_destinations):
                expr = LinExpr()
                for i in range(self.n_origins):
                    expr.add(self.x[(i, j)], 1.0)
                self.demand_constraints[j] = self.prob.addConstr(expr == demands[j], name=f"demand_{j}")
            # Supply constraints: <= supply
            for i in range(self.n_origins):
                expr = LinExpr()
                for j in range(self.n_destinations):
                    expr.add(self.x[(i, j)], 1.0)
                self.supply_constraints[i] = self.prob.addConstr(expr <= supplies[i], name=f"supply_{i}")
        else:
            # Supply constraints: == supply
            for i in range(self.n_origins):
                expr = LinExpr()
                for j in range(self.n_destinations):
                    expr.add(self.x[(i, j)], 1.0)
                self.supply_constraints[i] = self.prob.addConstr(expr == supplies[i], name=f"supply_{i}")
            # Demand constraints: <= demand
            for j in range(self.n_destinations):
                expr = LinExpr()
                for i in range(self.n_origins):
                    expr.add(self.x[(i, j)], 1.0)
                self.demand_constraints[j] = self.prob.addConstr(expr <= demands[j], name=f"demand_{j}")

        self.prob.update()
        return messages

    
    def solve(self):
        """
        Solves the transportation problem.

        Returns:
            bool: True if an optimal solution is found, False otherwise.
        """
        self.prob.optimize()
        self.status = self.prob.status
        self.status_message = self.get_status_message()
        return self.prob.status == GRB.OPTIMAL
    
    def get_status_message(self):
        """
        Get a human-readable status message based on the Gurobi status code.
        
        Returns:
            str: Status message describing the solution status
        """
        status_messages = {
            GRB.OPTIMAL: "Optimal solution found",
            GRB.INFEASIBLE: "Problem is infeasible - no feasible solution exists",
            GRB.INF_OR_UNBD: "Problem is unbounded or infeasible",
            GRB.UNBOUNDED: "Problem is unbounded - objective can decrease indefinitely",
            GRB.ITERATION_LIMIT: "Iteration limit reached before optimal solution found",
            GRB.NODE_LIMIT: "Node limit reached before optimal solution found",
            GRB.TIME_LIMIT: "Time limit reached before optimal solution found",
            GRB.SOLUTION_LIMIT: "Solution limit reached",
            GRB.INTERRUPTED: "Optimization was interrupted by the user",
            GRB.NUMERIC: "Numerical issues encountered during optimization",
            GRB.SUBOPTIMAL: "Suboptimal solution found",
            GRB.INPROGRESS: "Optimization in progress",
            GRB.USER_OBJ_LIMIT: "User objective limit reached"
        }
        return status_messages.get(self.status, f"Unknown status code: {self.status}")

    def get_solution(self):
        """
        Retrieves the solution of the transportation problem.

        Returns:
            dict: A dictionary containing the solution matrix, total cost, and flows.
        """
        if self.status != GRB.OPTIMAL:
            return {
                "solution": [],
                "cost": 0,
                "flows": [],
                "status": self.status,
                "status_message": self.status_message
            }
            
        solution = np.zeros((self.n_origins, self.n_destinations))
        for i in range(self.n_origins):
            for j in range(self.n_destinations):
                solution[i, j] = self.x[(i, j)].x
        cost = sum(self.costs[i][j] * solution[i, j] for i in range(self.n_origins) for j in range(self.n_destinations))
        flows = [{'from': i, 'to': j, 'quantity': solution[i, j], 'cost': self.costs[i][j],
                  'total_cost': self.costs[i][j] * solution[i, j]}
                 for i in range(self.n_origins) for j in range(self.n_destinations) if solution[i, j] > 1e-6]
        return {
            "solution": solution.tolist(), 
            "cost": cost, 
            "flows": flows,
            "status": self.status,
            "status_message": self.status_message
        }

    def is_degenerate(self):
        """
        Checks if the solution is degenerate.

        Returns:
            bool: True if the solution is degenerate, False otherwise.
        """
        if self.status != GRB.OPTIMAL:
            return False
            
        basic_vars = sum(1 for var in self.x.values() if abs(var.x) > 1e-6)
        return basic_vars < (self.n_origins + self.n_destinations - 1)

    def get_dual_values(self):
        """
        Retrieves the dual values of the constraints.

        Returns:
            dict: A dictionary containing dual values for supply and demand constraints.
        """
        if self.status != GRB.OPTIMAL:
            return None
            
        supply_duals = {i: self.supply_constraints[i].Pi for i in self.supply_constraints}
        demand_duals = {j: self.demand_constraints[j].Pi for j in self.demand_constraints}
        return {"supply_duals": supply_duals, "demand_duals": demand_duals}