import pandas as pd
from tabulate import tabulate
from typing import Dict, List, Union
from pathlib import Path
from markdown import markdown
from bs4 import BeautifulSoup

class TableParser:
    """Parser for converting paper summary tables to different formats with enhanced markdown support"""
    
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
        
    def _enhance_markdown(self, content: str) -> str:
        """Enhance markdown content with proper formatting"""
        if not content:
            return ""
            
        # Convert markdown to HTML
        html = markdown(content, extensions=[
            'extra',       # For tables, code blocks, etc.
            'nl2br',       # Convert newlines to <br>
            'sane_lists',  # Better list handling
            'fenced_code', # Code blocks
            'tables',      # Markdown tables
            'smarty',      # Smart quotes, dashes, etc.
        ])
        
        # Use BeautifulSoup to clean up and format the HTML
        soup = BeautifulSoup(html, 'html.parser')
        
        # Add CSS classes to elements
        for tag in soup.find_all(['p', 'ul', 'ol', 'li', 'pre', 'code']):
            if tag.name == 'p':
                tag['class'] = 'md-paragraph'
            elif tag.name in ['ul', 'ol']:
                tag['class'] = 'md-list'
            elif tag.name == 'li':
                tag['class'] = 'md-list-item'
            elif tag.name == 'pre':
                tag['class'] = 'md-code-block'
            elif tag.name == 'code':
                tag['class'] = 'md-inline-code'
                
        return str(soup)

    def to_html(self, file_path: Union[str, Path], title: str = "Paper Summaries") -> None:
        """Generate a standalone HTML page with enhanced markdown support"""
        # Convert markdown in cells to HTML
        html_data = []
        
        for row in self.table_data:
            html_row = {}
            for key, value in row.items():
                html_row[key] = self._enhance_markdown(value) if value else ""
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
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                /* Table styling */
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 25px 0;
                    font-size: 0.9em;
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                    min-width: 400px;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
                }}
                
                th, td {{
                    padding: 12px 15px;
                    border: 1px solid #ddd;
                    vertical-align: top;
                }}
                
                th {{
                    background-color: #009879;
                    color: #ffffff;
                    text-align: left;
                    position: sticky;
                    top: 0;
                }}
                
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                
                tr:hover {{
                    background-color: #f1f3f5;
                }}
                
                /* Markdown content styling */
                .md-paragraph {{
                    margin: 0.5em 0;
                    line-height: 1.6;
                }}
                
                .md-list {{
                    margin: 0.5em 0;
                    padding-left: 1.5em;
                }}
                
                .md-list-item {{
                    margin: 0.25em 0;
                }}
                
                .md-code-block {{
                    background-color: #f8f9fa;
                    padding: 0.75em;
                    border-radius: 4px;
                    overflow-x: auto;
                    margin: 0.5em 0;
                }}
                
                .md-inline-code {{
                    background-color: #f8f9fa;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                    font-size: 0.9em;
                }}
                
                /* Responsive design */
                @media (max-width: 768px) {{
                    table {{
                        display: block;
                        overflow-x: auto;
                    }}
                    
                    th, td {{
                        padding: 8px;
                        font-size: 0.85em;
                    }}
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            {html_table}
        </body>
        </html>
        """
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
