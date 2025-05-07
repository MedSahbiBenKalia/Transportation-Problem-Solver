from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model import TransportModel
import numpy as np

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
    dual_values: dict | None
    messages: List[str] | None  # Include warnings in the response

@app.post("/solve_transport", response_model=SolutionResponse)
async def solve_transport_problem(costs_data: Costs, supplies_data: Supplies, demands_data: Demands):
    costs = np.array(costs_data.costs)
    supplies = np.array(supplies_data.supplies)
    demands = np.array(demands_data.demands)

    try:
        messages = model_instance.build_model(costs, supplies, demands)
        if model_instance.solve():
            solution = model_instance.get_solution()
            dual_values = model_instance.get_dual_values()
            return {
                "solution": solution["solution"],
                "cost": solution["cost"],
                "flows": solution["flows"],
                "dual_values": dual_values,
                "messages": messages  # Include warnings in the response
            }
        else:
            raise HTTPException(status_code=400, detail="No optimal solution found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)