import typer

from .example import example

app = typer.Typer()
app.command(name="example")(example)


def main() -> None:
    app()
