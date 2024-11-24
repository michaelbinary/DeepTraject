# src/main.py
import asyncio
import logging
from rich.console import Console
from simulation.runner import SimulationRunner


async def main():
    """Main entry point for DeepTraject"""
    console = Console()

    try:
        console.print("[bold blue]Starting DeepTraject System[/bold blue]")

        # Initialize simulation
        runner = SimulationRunner()

        # Run benchmarks
        await runner.run_benchmarks()

        console.print("[bold green]Simulation completed successfully![/bold green]")

    except Exception as e:
        logging.exception("Error during simulation")
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        Console().print("[yellow]Simulation interrupted by user[/yellow]")
    except Exception as e:
        Console().print(f"[bold red]Error: {str(e)}[/bold red]")
        raise