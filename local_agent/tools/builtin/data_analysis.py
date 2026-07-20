"""
Data Analysis Tools - pandas, matplotlib, statistics
"""
import json
from pathlib import Path
import tempfile
from typing import Optional
from local_agent.core.tools import tool


@tool
def data_load_csv(file_path: str, separator: str = ",") -> str:
    """
    Load a CSV file and return summary information.
    Args:
        file_path: Path to the CSV file
        separator: Column separator (default: comma)
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path, sep=separator)
        info_parts = [
            f"Loaded CSV: {file_path}",
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            f"\nData types:\n{df.dtypes.to_string()}",
            f"\nFirst 5 rows:\n{df.head().to_string()}",
            f"\nMissing values:\n{df.isnull().sum().to_string()}",
        ]
        return "\n".join(info_parts)
    except ImportError:
        return "Error: pandas not installed. Run: pip install pandas"
    except Exception as e:
        return f"Error loading CSV: {e}"


@tool
def data_describe(file_path: str, columns: str = "") -> str:
    """
    Get statistical description of a CSV dataset.
    Args:
        file_path: Path to the CSV file
        columns: Comma-separated column names to analyze (empty = all numeric columns)
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        if columns:
            col_list = [c.strip() for c in columns.split(",")]
            df = df[col_list]
        return f"Statistical summary of {file_path}:\n\n{df.describe().to_string()}"
    except Exception as e:
        return f"Error: {e}"


@tool
def data_query(file_path: str, query: str) -> str:
    """
    Query a CSV file using pandas query syntax or Python expression.
    Args:
        file_path: Path to the CSV file
        query: Pandas query string (e.g., "age > 30 and salary > 50000")
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        result = df.query(query)
        return (
            f"Query: {query}\n"
            f"Results: {len(result)} rows\n\n"
            f"{result.head(20).to_string()}"
        )
    except Exception as e:
        return f"Error querying data: {e}"


@tool
def data_run_analysis(file_path: str, analysis_code: str) -> str:
    """
    Run custom pandas analysis code on a CSV file.
    The dataframe is available as 'df'.
    Args:
        file_path: Path to the CSV file
        analysis_code: Python code using 'df' as the dataframe variable
    """
    import subprocess
    import sys
    import tempfile
    import os

    code = f"""
import pandas as pd
import numpy as np
df = pd.read_csv("{file_path}")
{analysis_code}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        return output[:5000] if output else "(No output)"
    except subprocess.TimeoutExpired:
        return "Analysis timed out"
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


@tool
def data_visualize(
    file_path: str,
    chart_type: str,
    x_column: str,
    y_column: str = "",
    title: str = "",
    output_path: Optional[str] = None,
) -> str:
    """
    Create a visualization from a CSV file.
    Args:
        file_path: Path to the CSV file
        chart_type: Type of chart: 'bar', 'line', 'scatter', 'histogram', 'pie', 'box'
        x_column: Column name for X axis (or data column for histogram/pie)
        y_column: Column name for Y axis (optional for some chart types)
        title: Chart title
        output_path: Where to save the chart image. Defaults to the system
            temporary directory when omitted.
    """
    try:
        output_path = output_path or str(Path(tempfile.gettempdir()) / "chart.png")
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        df = pd.read_csv(file_path)
        fig, ax = plt.subplots(figsize=(10, 6))

        chart_title = title or f"{chart_type.title()} Chart"
        ax.set_title(chart_title)

        if chart_type == "bar":
            if y_column and y_column in df.columns:
                df.plot(kind="bar", x=x_column, y=y_column, ax=ax)
            else:
                df[x_column].value_counts().plot(kind="bar", ax=ax)
        elif chart_type == "line":
            if y_column:
                ax.plot(df[x_column], df[y_column])
                ax.set_xlabel(x_column)
                ax.set_ylabel(y_column)
            else:
                df[x_column].plot(ax=ax)
        elif chart_type == "scatter":
            ax.scatter(df[x_column], df[y_column])
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
        elif chart_type == "histogram":
            df[x_column].hist(ax=ax, bins=20)
            ax.set_xlabel(x_column)
            ax.set_ylabel("Frequency")
        elif chart_type == "pie":
            df[x_column].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
        elif chart_type == "box":
            if y_column:
                df.boxplot(column=y_column, by=x_column, ax=ax)
            else:
                df[[x_column]].boxplot(ax=ax)
        else:
            return f"Unknown chart type: {chart_type}. Use: bar, line, scatter, histogram, pie, box"

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()
        return f"Chart saved to: {output_path}"
    except ImportError:
        return "Error: pandas/matplotlib not installed"
    except Exception as e:
        return f"Error creating chart: {e}"


@tool
def data_to_json(file_path: str, orient: str = "records", max_rows: int = 100) -> str:
    """Convert CSV data to JSON format."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        if len(df) > max_rows:
            df = df.head(max_rows)
            suffix = f"\n(Showing first {max_rows} rows)"
        else:
            suffix = ""
        return df.to_json(orient=orient, indent=2) + suffix
    except Exception as e:
        return f"Error: {e}"


@tool
def data_merge_csv(file1: str, file2: str, on_column: str, how: str = "inner", output_path: Optional[str] = None) -> str:
    """
    Merge two CSV files on a common column.
    Args:
        file1: Path to first CSV file
        file2: Path to second CSV file
        on_column: Column name to merge on
        how: Merge type: 'inner', 'left', 'right', 'outer'
        output_path: Where to save the merged CSV. Defaults to the system
            temporary directory when omitted.
    """
    try:
        output_path = output_path or str(Path(tempfile.gettempdir()) / "merged.csv")
        import pandas as pd
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        merged = pd.merge(df1, df2, on=on_column, how=how)
        merged.to_csv(output_path, index=False)
        return (
            f"Merged {file1} and {file2} on '{on_column}' ({how} join)\n"
            f"Result: {merged.shape[0]} rows × {merged.shape[1]} columns\n"
            f"Saved to: {output_path}"
        )
    except Exception as e:
        return f"Error merging CSV files: {e}"


data_load_csv.metadata = data_load_csv.metadata or {}
data_load_csv.metadata["category"] = "data"
data_describe.metadata = data_describe.metadata or {}
data_describe.metadata["category"] = "data"
data_query.metadata = data_query.metadata or {}
data_query.metadata["category"] = "data"
data_run_analysis.metadata = data_run_analysis.metadata or {}
data_run_analysis.metadata["category"] = "data"
data_visualize.metadata = data_visualize.metadata or {}
data_visualize.metadata["category"] = "data"
data_to_json.metadata = data_to_json.metadata or {}
data_to_json.metadata["category"] = "data"
data_merge_csv.metadata = data_merge_csv.metadata or {}
data_merge_csv.metadata["category"] = "data"

TOOLS = [
    data_load_csv,
    data_describe,
    data_query,
    data_run_analysis,
    data_visualize,
    data_to_json,
    data_merge_csv,
]
