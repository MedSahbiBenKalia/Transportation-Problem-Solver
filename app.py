from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from model import TransportModel
import numpy as np
from gurobipy import GRB

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend URL, e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

model_instance = TransportModel()

class Costs(BaseModel):
    costs: List[List[float]]

class Supplies(BaseModel):
    supplies: List[float]

class Demands(BaseModel):
    demands: List[float]

class SolutionResponse(BaseModel):
    solution: List[List[float]]
    cost: float
    flows: List[dict]
    dual_values: Optional[dict] = None
    messages: Optional[List[str]] = None  # Include warnings in the response
    status: int
    status_message: str
    is_optimal: bool

@app.post("/solve_transport", response_model=SolutionResponse)
async def solve_transport_problem(costs_data: Costs, supplies_data: Supplies, demands_data: Demands):
    print("Received data for solving the transportation problem")
    print(f"Costs: {costs_data.costs}")
    print(f"Supplies: {supplies_data.supplies}")
    print(f"Demands: {demands_data.demands}")
    costs = np.array(costs_data.costs)
    supplies = np.array(supplies_data.supplies)
    demands = np.array(demands_data.demands)

    try:
        messages = model_instance.build_model(costs, supplies, demands)
        is_optimal = model_instance.solve()
        solution = model_instance.get_solution()
        dual_values = model_instance.get_dual_values() if is_optimal else None
        
        return {
            "solution": solution["solution"],
            "cost": solution["cost"],
            "flows": solution["flows"],
            "dual_values": dual_values,
            "messages": messages,
            "status": solution["status"],
            "status_message": solution["status_message"],
            "is_optimal": is_optimal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status_codes")
async def get_status_codes():
    """
    Returns a dictionary of Gurobi status codes and their meanings
    """
    status_codes = {
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
    return {"status_codes": status_codes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)