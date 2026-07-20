"""Server CLI command"""
import typer
from rich.console import Console

app = typer.Typer(help="Start the LocalAgent web server")
console = Console()


@app.command("start")
def server_start(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the LocalAgent web server"""
    console.print(f"[bold cyan]🚀 Starting LocalAgent Server[/bold cyan]")
    console.print(f"[dim]Address: http://{host}:{port}[/dim]")
    import uvicorn
    uvicorn.run("local_agent.api.app:app", host=host, port=port, reload=reload)
