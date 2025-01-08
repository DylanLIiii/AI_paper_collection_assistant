import pandas as pd
from tabulate import tabulate
import mistune
from typing import Dict, List, Union
from pathlib import Path

class TableParser:
    """Parser for converting paper summary tables to different formats"""
    
    def __init__(self, table_data: List[Dict[str, str]]):
        """
        Initialize with table data
        
        Args:
            table_data: List of dictionaries where each dict represents a row
        """
        self.table_data = table_data
        self.columns = list(table_data[0].keys()) if table_data else []
        
    def to_markdown(self) -> str:
        """Convert table data to markdown format"""
        return tabulate(
            self.table_data,
            headers="keys",
            tablefmt="github",  # GitHub-style markdown
            showindex=False
        )
    
    def to_excel(self, file_path: Union[str, Path]) -> None:
        """Save table data to Excel file"""
        df = pd.DataFrame(self.table_data)
        df.to_excel(file_path, index=False)
        
    def to_html(self, file_path: Union[str, Path], title: str = "Paper Summaries") -> None:
        """Generate a standalone HTML page with markdown parsed in cells"""
        # Convert markdown in cells to HTML
        html_data = []
        markdown = mistune.create_markdown(escape=False)
        
        for row in self.table_data:
            html_row = {}
            for key, value in row.items():
                html_row[key] = markdown(value) if value else ""
            html_data.append(html_row)
            
        # Generate HTML table
        html_table = tabulate(
            html_data,
            headers="keys",
            tablefmt="unsafehtml",
            showindex=False
        )
        
        # Create full HTML page
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 25px 0;
                    font-size: 0.9em;
                    font-family: sans-serif;
                    min-width: 400px;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                }}
                th, td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                }}
                th {{
                    background-color: #009879;
                    color: #ffffff;
                    text-align: left;
                }}
                tr:nth-child(even) {{
                    background-color: #f3f3f3;
                }}
                tr:hover {{
                    background-color: #f1f1f1;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {html_table}
        </body>
        </html>
        """
        
        with open(file_path, "w") as f:
            f.write(html_content)
