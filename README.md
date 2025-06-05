# Math Agent

A LangGraph-based agent that can perform mathematical operations using a set of tools.

## Features

- Mathematical operations:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - Modulus
- Powered by LangGraph and LangChain
- Uses OpenAI's GPT-4 for natural language understanding

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the agent:
```bash
python agent.py
```

The agent can handle natural language queries about mathematical operations. For example:
- "What is 5.5 + 3.2?"
- "Calculate 10 * 5 - 3"
- "What's the remainder when 17 is divided by 5?"

## Project Structure

- `agent.py`: Main agent implementation with mathematical tools
- `app.py`: Web application interface
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not tracked in git)

## Development

The agent is built using:
- LangGraph for the agent workflow
- LangChain for LLM integration
- OpenAI's GPT-4 for natural language processing

## License

[Your chosen license]
