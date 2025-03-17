# CSV Question Answering & Visualization App

## Overview
This application enables users to upload CSV files, ask natural language questions about the data, and visualize relationships between variables through plots. It combines the power of large language models (LLM) with data analysis and visualization capabilities.

## Features
- **CSV File Analysis**: Upload and analyze CSV files up to 25MB
- **Natural Language Questions**: Ask questions about your data in plain English
- **AI-Powered Answers**: Get detailed responses using LLM technology
- **Data Visualization**: Generate plots to visualize relationships between variables
- **Structured Analysis**: Receive statistical summaries and insights

## Technical Architecture
The application consists of several interconnected modules:

1. **Pydantic AI Models**: Structured data models with AI capabilities
2. **CSV Handler**: Parses and processes CSV files
3. **Visualization Module**: Generates plots based on selected columns
4. **Main Application Logic**: Integrates all components
5. **Gradio Interface**: Provides a user-friendly web interface

## Requirements
- Python 3.8+
- Gradio
- Pandas
- Matplotlib
- Ollama
- Pydantic
- PydanticAI

## Installation

```bash


# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install gradio pandas matplotlib ollama pydantic pydanticai
```

## Setup

1. Make sure you have Ollama installed and the LLama 3.1 model downloaded:
```bash
ollama pull llama3.1
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

## Usage

1. **Upload a CSV file**: Click the file upload area to select a CSV file from your computer
2. **Ask a question**: Type your question about the data in the text field
3. **Visualization**: If you want to create a plot, enter the column names for X and Y axes
4. **Submit**: Click the "Submit" button to get your answer and visualization

### Example Questions
- "What is the average price of houses in this dataset?"
- "Is there a correlation between square footage and price?"
- "What are the top 5 most expensive items?"
- "How does the price vary by location?"

## How It Works

1. The application parses the uploaded CSV file using Pandas
2. It generates a summary of the data including columns, data types, and sample rows
3. Your question is sent to the LLama 3.1 model through Ollama along with the data summary
4. The model generates a detailed answer based on the data
5. If visualization columns are provided, a plot is generated using Matplotlib

## Customization

### Changing the LLM Model
You can modify the `model_name` in the `CSVAnalysisQuery` class to use a different Ollama model.

### Extending Visualization Types
The `Visualizer` class can be extended to support additional visualization types like bar charts, scatter plots, etc.

## Limitations
- The application works best with well-structured CSV files
- Very large files (>25MB) may cause performance issues
- The quality of answers depends on the LLM model used
- Complex visualizations may require additional customization

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT License](LICENSE)
