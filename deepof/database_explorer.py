import duckdb
import os
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

class DuckDBExplorer:
    def __init__(self):
        # --- UI Components ---
        self.db_path_input = widgets.Text(
            value='',
            placeholder='Paste your DuckDB file path here...',
            description='DB Path:',
            layout=widgets.Layout(width='80%')
        )

        self.load_button = widgets.Button(description="Load Tables", button_style='primary', layout=widgets.Layout(width='150px'))
        self.table_dropdown = widgets.Dropdown(description="Table:", layout=widgets.Layout(width='80%'))

        self.query_area = widgets.Textarea(
            description="SQL Query:",
            layout=widgets.Layout(width='100%', height='150px')
        )

        self.execute_button = widgets.Button(
            description="Execute", button_style='success', layout=widgets.Layout(width='150px')
        )

        self.output_area = widgets.Output()

        # Bind actions
        self.load_button.on_click(self.load_tables)
        self.table_dropdown.observe(self.on_table_change, names='value')
        self.execute_button.on_click(self.execute_query)

    def display_ui(self):
        """Call this method to show the Explorer UI in a Jupyter Notebook."""
        display(widgets.VBox([
            widgets.HBox([self.db_path_input, self.load_button]),
            widgets.HBox([self.table_dropdown]),
            self.query_area,
            self.execute_button,
            self.output_area
        ]))

    def load_tables(self, b):
        path = self.db_path_input.value.strip()
        if os.path.exists(path):
            try:
                with duckdb.connect(path) as conn:
                    tables = conn.execute("SHOW TABLES").fetchall()
                    table_names = [t[0] for t in tables]
                    if table_names:
                        self.table_dropdown.options = table_names
                        self.table_dropdown.value = table_names[0]
                        self.run_default_query(table_names[0])
                    else:
                        self.table_dropdown.options = []
                        self.query_area.value = ''
                        with self.output_area:
                            clear_output()
                            display(HTML("<b>No tables found in the database.</b>"))
            except Exception as e:
                with self.output_area:
                    clear_output()
                    display(HTML(f"<b style='color:red;'>Error loading tables: {e}</b>"))
        else:
            with self.output_area:
                clear_output()
                display(HTML("<b style='color:red;'>⚠️ Invalid DB path. Please check the file location.</b>"))

    def run_default_query(self, table_name):
        default_query = f"SELECT * FROM {table_name} LIMIT 20"
        self.query_area.value = default_query
        self.execute_query(query=default_query)

    def execute_query(self, b=None, query=None):
        path = self.db_path_input.value.strip()
        sql = query if query else self.query_area.value.strip()

        if not sql:
            with self.output_area:
                clear_output()
                display(HTML("<b style='color:red;'>⚠️ Query is empty.</b>"))
            return

        if not sql.lower().startswith("select"):
            with self.output_area:
                clear_output()
                display(HTML("<b style='color:red;'>⚠️ Only SELECT queries are allowed.</b>"))
            return

        try:
            with duckdb.connect(path) as conn:
                df = conn.execute(sql).fetchdf()
                with self.output_area:
                    clear_output()

                    table_html = df.to_html(index=False)

                    scrollable_html = f"""
                    <div style="
                        overflow-x: auto; 
                        overflow-y: auto; 
                        max-height: 400px; 
                        max-width: 100%; 
                        border: 1px solid #ccc; 
                        padding: 5px;
                        white-space: nowrap;
                    ">
                        {table_html}
                    </div>
                    """

                    display(HTML(scrollable_html))
        except Exception as e:
            with self.output_area:
                clear_output()
                display(HTML(f"<b style='color:red;'>Error executing query: {e}</b>"))

    def on_table_change(self, change):
        selected_table = change['new']
        if selected_table:
            self.run_default_query(selected_table)
