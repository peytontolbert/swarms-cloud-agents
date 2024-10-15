from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from swarms import Agent
from swarm_models import OpenAIChat
from swarms_memory import ChromaDB
import subprocess
import os

app = FastAPI()

# Function to instantiate an agent from user configuration
def create_agent(agent_config):

    # Initialize model
    model = OpenAIChat(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=agent_config.get('model_name', "gpt-4o-mini"),
        temperature=agent_config.get('temperature', 0.1),
    )

    # Initialize agent
    agent = Agent(
        agent_name=agent_config.get('agent_name', 'Devin'),
        system_prompt=agent_config.get('system_prompt', 'Autonomous agent that can interact with humans.'),
        llm=model,
        max_loops=agent_config.get('max_loops', '2'),
        autosave=agent_config.get('autosave', True),
        verbose=agent_config.get('verbose', True),
    )

    return agent
    
# Define a Pydantic model for the request body
class AgentRequest(BaseModel):
    agent_config: dict
    query: str

# API route to receive agent config and run the agent
@app.post("/run_agent")
def run_agent(request: AgentRequest):
    try:
        agent = create_agent(request.agent_config)
        result = agent.run(request.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
