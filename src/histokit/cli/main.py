import typer

from .example import example
from .thumbnail import thumbnail

app = typer.Typer()
app.command(name="example")(example)
app.command(name="thumbnail")(thumbnail)


def main() -> None:
    print("Welcome to Histokit CLI!")
    print("Use --help to see available commands.")
    app()
