import sys
import time
import subprocess
import os
from fastapi import FastAPI, HTTPException
import uvicorn
from typing import Dict, Any, Literal, Union
from pydantic import BaseModel
from swarms import Agent
from swarms.prompts.tools import tool_sop_prompt
from swarms_memory import BaseVectorDatabase
from swarm_models import OpenAIChat
from swarm_models.tiktoken_wrapper import TikTokenizer
from swarms_memory import ChromaDB
import asyncio
app = FastAPI()

agent_output_type = Literal[
    "string", "str", "list", "json", "dict", "yaml"
]
ToolUsageType = Union[BaseModel, Dict[str, Any]]

# Define a Pydantic model for the request body
class AgentRequest(BaseModel):
    agent_config: dict
    query: str

# API route to receive agent config and run the agent
@app.post("/run_agent")
async def run_agent(request: AgentRequest):
    try:
        agent = create_agent(request.agent_config)
        result = await asyncio.wait_for(agent.run(request.query), timeout=60)
        return {"response": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")
    except Exception as e:
        # Log the error internally
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

def create_agent(agent_config):
    agent = Agent(
        agent_id=agent_config.get('agent_id', None),
        id=agent_config.get('id', None),
        llm=agent_config.get('llm', None),
        template=agent_config.get('template', None),
        max_loops=agent_config.get('max_loops', 2),
        stopping_condition=agent_config.get('stopping_condition', None),
        loop_interval=agent_config.get('loop_interval', 0),
        retry_attempts=agent_config.get('retry_attempts', 3),
        retry_interval=agent_config.get('retry_interval', 1),
        return_history=agent_config.get('return_history', False),
        stopping_token=agent_config.get('stopping_token', None),
        dynamic_loops=agent_config.get('dynamic_loops', False),
        interactive=agent_config.get('interactive', False),
        dashboard=agent_config.get('dashboard', False),
        agent_name=agent_config.get('agent_name', 'Devin'),
        agent_description=agent_config.get('agent_description', None),
        system_prompt=agent_config.get('system_prompt', 'Autonomous agent that can interact with humans.'),
        tools=agent_config.get('tools', []),
        dynamic_temperature_enabled=agent_config.get('dynamic_temperature_enabled', False),
        sop=agent_config.get('sop', None),
        sop_list=agent_config.get('sop_list', None),
        saved_state_path=agent_config.get('saved_state_path', None),
        autosave=agent_config.get('autosave', True),
        context_length=agent_config.get('context_length', 8192),
        user_name=agent_config.get('user_name', None),
        self_healing_enabled=agent_config.get('self_healing_enabled', False),
        code_interpreter=agent_config.get('code_interpreter', False),
        multi_modal=agent_config.get('multi_modal', None),
        pdf_path=agent_config.get('pdf_path', None),
        list_of_pdf=agent_config.get('list_of_pdf', None),
        tokenizer=agent_config.get('tokenizer', TikTokenizer()),
        long_term_memory=agent_config.get('long_term_memory', None),
        preset_stopping_token=agent_config.get('preset_stopping_token', False),
        traceback=agent_config.get('traceback', None),
        traceback_handlers=agent_config.get('traceback_handlers', None),
        streaming_on=agent_config.get('streaming_on', False),
        docs=agent_config.get('docs', []),
        docs_folder=agent_config.get('docs_folder', None),
        verbose=agent_config.get('verbose', None),
        parser=agent_config.get('parser', None),
        best_of_n=agent_config.get('best_of_n', None),
        callback=agent_config.get('callback', None),
        metadata=agent_config.get('metadata', None),
        callbacks=agent_config.get('callbacks', None),
        logger_handler=agent_config.get('logger_handler', None),
        search_algorithm=agent_config.get('search_algorithm', None),
        logs_to_filename=agent_config.get('logs_to_filename', None),
        evaluator=agent_config.get('evaluator', None),
        stopping_func=agent_config.get('stopping_func', None),
        custom_loop_condition=agent_config.get('custom_loop_condition', None),
        sentiment_threshold=agent_config.get('sentiment_threshold', None),
        custom_exit_command=agent_config.get('custom_exit_command', "exit"),
        sentiment_analyzer=agent_config.get('sentiment_analyzer', None),
        limit_tokens_from_string=agent_config.get('limit_tokens_from_string', None),
        custom_tools_prompt=agent_config.get('custom_tools_prompt', None),
        tool_schema=agent_config.get('tool_schema', None),
        output_type=agent_config.get('output_type', "str"),
        function_calling_type=agent_config.get('function_calling_type', "json"),
        output_cleaner=agent_config.get('output_cleaner', None),
        function_calling_format_type=agent_config.get('function_calling_format_type', "OpenAI"),
        list_base_models=agent_config.get('list_base_models', None),
        metadata_output_type=agent_config.get('metadata_output_type', "json"),
        state_save_file_type=agent_config.get('state_save_file_type', "json"),
        chain_of_thoughts=agent_config.get('chain_of_thoughts', False),
        algorithm_of_thoughts=agent_config.get('algorithm_of_thoughts', False),
        tree_of_thoughts=agent_config.get('tree_of_thoughts', False),
        tool_choice=agent_config.get('tool_choice', "auto"),
        execute_tool=agent_config.get('execute_tool', False),
        rules=agent_config.get('rules', None),
        planning=agent_config.get('planning', False),
        planning_prompt=agent_config.get('planning_prompt', None),
        device=agent_config.get('device', None),
        custom_planning_prompt=agent_config.get('custom_planning_prompt', None),
        memory_chunk_size=agent_config.get('memory_chunk_size', 2000),
        agent_ops_on=agent_config.get('agent_ops_on', False),
        log_directory=agent_config.get('log_directory', None),
        tool_system_prompt=agent_config.get('tool_system_prompt', None),
        max_tokens=agent_config.get('max_tokens', 4096),
        top_p=agent_config.get('top_p', 0.9),
        top_k=agent_config.get('top_k', None),
        frequency_penalty=agent_config.get('frequency_penalty', 0.0),
        presence_penalty=agent_config.get('presence_penalty', 0.0),
        temperature=agent_config.get('temperature', 0.1),
        workspace_dir=agent_config.get('workspace_dir', "agent_workspace"),
        timeout=agent_config.get('timeout', None),
        created_at=agent_config.get('created_at', time.time()),
        return_step_meta=agent_config.get('return_step_meta', False),
        tags=agent_config.get('tags', None),
        use_cases=agent_config.get('use_cases', None),
        step_pool=agent_config.get('step_pool', []),
        print_every_step=agent_config.get('print_every_step', False),
        time_created=agent_config.get('time_created', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
        agent_output=agent_config.get('agent_output', None),
        executor_workers=agent_config.get('executor_workers', os.cpu_count()),
        data_memory=agent_config.get('data_memory', None),
        load_yaml_path=agent_config.get('load_yaml_path', None),
        # ... include other necessary parameters based on swarms.Agent's constructor ...
    )
    return agent

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)
