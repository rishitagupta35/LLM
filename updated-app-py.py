import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import ollama
import io
from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field
from pydantic_ai import AIModel, OpenAISchema, AIBaseModel

# ===== 1. PYDANTIC AI MODELS =====
class CSVAnalysisQuery(AIModel):
    """Model for structuring CSV analysis queries with AI capabilities."""
    question: str = Field(..., description="The question about the CSV data")
    csv_summary: str = Field(..., description="Summary of the CSV data including column names and first few rows")
    
    class Config:
        # Configure with Ollama as the LLM backend
        model_name = "llama3.1"
        provider = "ollama"
    
    def execute(self) -> str:
        """Execute the query using the Ollama LLM through PydanticAI."""
        try:
            prompt = f"""
            Analyze this CSV data summary:
            
            {self.csv_summary}
            
            Question: {self.question}
            
            Provide a clear, detailed answer based only on the data provided. 
            Include relevant statistics when appropriate.
            """
            
            # Using Ollama directly since PydanticAI's direct integration with Ollama 
            # may require specific setup
            response = ollama.chat(model=self.Config.model_name, messages=[{
                "role": "user",
                "content": prompt
            }])
            
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            return "Error: Unexpected response format from LLM."
        except Exception as e:
            return f"LLM processing error: {str(e)}"


class ColumnAnalysisResult(AIBaseModel):
    """Structured result for column analysis."""
    column_name: str = Field(..., description="Name of the analyzed column")
    data_type: str = Field(..., description="Data type of the column")
    statistical_summary: str = Field(..., description="Statistical summary of the column values")
    observations: str = Field(..., description="Key observations about the column")


class DataAnalysisResult(AIBaseModel):
    """Structured output for data analysis."""
    answer: str = Field(..., description="Direct answer to the user's question")
    reasoning: str = Field(..., description="Reasoning process to reach the answer")
    relevant_columns: List[str] = Field(default=[], description="Columns that were most relevant to the question")
    suggested_visualizations: Optional[List[str]] = Field(default=None, description="Suggested visualizations that might help")


class GraphRequest(AIBaseModel):
    """Model for graph generation requests."""
    x_column: str = Field(..., description="Column name for X-axis")
    y_column: str = Field(..., description="Column name for Y-axis")
    
    def validate_columns(self, df: pd.DataFrame) -> Optional[str]:
        """Validate if the specified columns exist in the dataframe."""
        missing_columns = []
        if self.x_column not in df.columns:
            missing_columns.append(self.x_column)
        if self.y_column not in df.columns:
            missing_columns.append(self.y_column)
            
        if missing_columns:
            return f"Error: Columns not found in CSV: {', '.join(missing_columns)}"
        return None


# ===== 2. FILE HANDLING MODULE =====
class CSVHandler:
    """Module for handling CSV file operations."""
    
    @staticmethod
    def parse_csv(file_bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Parse CSV file from uploaded bytes."""
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            return df, None
        except Exception as e:
            return None, f"CSV parsing error: {str(e)}"
    
    @staticmethod
    def get_csv_summary(df: pd.DataFrame) -> str:
        """Generate a summary of the CSV data."""
        summary = f"CSV Columns: {', '.join(df.columns)}\n\n"
        summary += f"First 5 rows:\n{df.head(5).to_string()}\n\n"
        summary += f"Data types:\n{df.dtypes.to_string()}\n\n"
        summary += f"Basic statistics for numeric columns:\n{df.describe().to_string()}"
        return summary


# ===== 3. VISUALIZATION MODULE =====
class Visualizer:
    """Module for generating data visualizations."""
    
    @staticmethod
    def generate_plot(df: pd.DataFrame, graph_request: GraphRequest) -> Tuple[Optional[str], Optional[str]]:
        """Generate a plot based on the specified columns."""
        # Validate columns first
        error = graph_request.validate_columns(df)
        if error:
            return None, error
        
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(df[graph_request.x_column], df[graph_request.y_column], marker='o', linestyle='-')
            plt.xlabel(graph_request.x_column)
            plt.ylabel(graph_request.y_column)
            plt.title(f"{graph_request.y_column} vs {graph_request.x_column}")
            plt.tight_layout()
            
            # Save plot to file
            plot_path = "plot.png"
            plt.savefig(plot_path)
            plt.close()
            return plot_path, None
        except Exception as e:
            return None, f"Plot generation error: {str(e)}"


# ===== 4. MAIN APPLICATION LOGIC =====
class CSVQAApplication:
    """Main application class that integrates all components."""
    
    @staticmethod
    def process_query(file, question, x_column, y_column):
        """Process user query and generate response with visualization."""
        # Parse CSV
        csv_handler = CSVHandler()
        df, csv_error = csv_handler.parse_csv(file)
        if csv_error:
            return csv_error, None
            
        # Generate LLM response using PydanticAI
        csv_summary = csv_handler.get_csv_summary(df)
        query = CSVAnalysisQuery(question=question, csv_summary=csv_summary)
        llm_answer = query.execute()
        
        # Generate visualization if columns are provided
        plot_path = None
        plot_error = None
        if x_column and y_column:
            graph_request = GraphRequest(x_column=x_column, y_column=y_column)
            visualizer = Visualizer()
            plot_path, plot_error = visualizer.generate_plot(df, graph_request)
            
        # Return the results
        if plot_error:
            return f"{llm_answer}\n\nVisualization Error: {plot_error}", None
        return llm_answer, plot_path


# ===== 5. GRADIO INTERFACE =====
def launch_app():
    """Launch the Gradio application."""
    with gr.Blocks() as app:
        gr.Markdown("# 📊 CSV Question Answering & Visualization App")
        gr.Markdown("Upload a CSV file, ask questions about the data, and visualize it with graphs.")
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload CSV File (Max 25MB)", type="binary")
                question_input = gr.Textbox(label="Ask a Question About the CSV", 
                                          placeholder="Example: What is the average price? Is there a correlation between X and Y?")
                
                with gr.Row():
                    col_x_input = gr.Textbox(label="X-axis Column Name", placeholder="Enter column name for X-axis")
                    col_y_input = gr.Textbox(label="Y-axis Column Name", placeholder="Enter column name for Y-axis")
                
                submit_button = gr.Button("Submit", variant="primary")
            
            with gr.Column():
                answer_output = gr.Textbox(label="Analysis Results", lines=10)
                plot_output = gr.Image(label="Data Visualization")
        
        submit_button.click(
            fn=CSVQAApplication.process_query,
            inputs=[file_input, question_input, col_x_input, col_y_input],
            outputs=[answer_output, plot_output]
        )
        
        # Examples for user guidance
        gr.Examples(
            examples=[
                ["Housing.csv", "What is the average price of houses in this dataset?", "area", "price"],
                ["Housing.csv", "Is there a correlation between square footage and price?", "bedrooms", "price"]
            ],
            inputs=[file_input, question_input, col_x_input, col_y_input]
        )
        
    return app


# ===== 6. APPLICATION ENTRY POINT =====
if __name__ == "__main__":
    app = launch_app()
    app.launch()
