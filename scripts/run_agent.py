"""CLI entrypoint to interact with the AI Customer Support Agent."""

import sys
import argparse
from pathlib import Path
import json
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.agent import CustomerSupportAgent


def main():
    parser = argparse.ArgumentParser(description="AI Customer Support Agent CLI")
    parser.add_argument("--message", "--query", "-m", "-q", dest="message", type=str, help="Incoming customer inquiry text")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON format")
    args = parser.parse_args()

    console = Console()
    config = load_config()
    
    agent = CustomerSupportAgent(config=config)

    if not args.message:
        console.print("[bold green]Interactive Customer Support Agent Mode[/bold green] (type 'exit' to quit)\n")
        while True:
            try:
                user_msg = console.input("[bold cyan]Customer Message > [/bold cyan]")
                if user_msg.strip().lower() in ["exit", "quit", "q"]:
                    break
                if not user_msg.strip():
                    continue

                resp = agent.run(user_msg)
                _display_response(console, user_msg, resp, raw_json=args.json)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        resp = agent.run(args.message)
        _display_response(console, args.message, resp, raw_json=args.json)


def _display_response(console: Console, message: str, resp, raw_json: bool = False):
    if raw_json:
        print(resp.model_dump_json(indent=2))
        return

    decision_color = "green" if resp.decision == "AUTO_HANDLE" else "bold red"
    
    panel_content = f"""
[bold]Classified Intent:[/bold] {resp.intent} (Confidence: {resp.confidence:.2%})
[bold]Decision:[/bold] [{decision_color}]{resp.decision}[/{decision_color}]
[bold]Reason:[/bold] {resp.reason}

[bold]Generated Support Reply:[/bold]
"{resp.reply}"

[bold]Historical Evidence Used:[/bold] {len(resp.evidence)} support pairs retrieved
[bold]Total Latency:[/bold] {resp.metadata.get('total_latency_ms', 0):.1f} ms
"""
    console.print(Panel(panel_content, title=f"AI Agent Response ({resp.intent})", expand=False))


if __name__ == "__main__":
    main()
