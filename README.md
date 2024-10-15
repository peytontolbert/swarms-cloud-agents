# Swarms Cloud Agents API

## Overview

This project provides an autoscaling infrastructure for hosting intelligent Swarms Agent API. It leverages FastAPI for API endpoints, integrates with OpenAI for language models, and utilizes Docker for containerization. The system is designed to automatically scale agent instances based on demand, ensuring efficient resource usage and optimal performance.

## Features

- **Autoscaling**: Automatically adjust the number of agent instances based on workload.
- **FastAPI Integration**: Robust and high-performance API framework for managing agents.
- **Persistent Memory**: Utilize ChromaDB for efficient memory management and retrieval.
- **Customizable Agents**: Configure agents with various settings to suit different tasks.
- **Docker Support**: Containerize the application for easy deployment and scalability.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- OpenAI API Key

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/peytontolbert/swarms-cloud-agents.git
   cd swarms-cloud-agents   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key   ```

4. **Build Docker Image**
   ```bash
   docker build -t autoscaling-agent .   ```

5. **Run the Application**
   ```bash
   docker run -d -p 5000:5000 --env-file .env autoscaling-agent   ```

## Usage

API endpoint to run an agent:

- **POST** `/run_agent`

  **Request Body:**
  ```json
  {
    "agent_config": {
      "agent_name": "Devin",
      "model_name": "gpt-4o-mini",
      "temperature": 0.1,
      "max_loops": 2,
      "autosave": true,
      "verbose": true
    },
    "query": "Your query here"
  }  ```

  **Response:**
  ```json
  {
    "response": "Agent's response..."
  }  ```

## Project Structure

- `main.py`: Entry point for the FastAPI application.
- `dockerfile`: Docker configuration for containerizing the application.
- `requirements.txt`: Python dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).

## Future Features

- Enhanced autoscaling algorithms for better efficiency.
- Additional agent capabilities and integrations.
- Improved monitoring and logging features.
- User authentication and authorization.
- Comprehensive testing and CI/CD pipelines.

