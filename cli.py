"""Pylox Systems — fine-tune pipeline CLI.

Single entry point for client engagements. Every command is idempotent + resumable.

Commands:
  onboard      full pipeline: intake -> enrich -> train -> eval -> deploy
  intake       validate + format client data
  enrich       run gpt-oss-120b enrichment on prepared data
  train        fine-tune LoRA on enriched data
  eval         generate eval report for a trained adapter
  deploy       push adapter + serve (runpod | spark)
  refresh      re-train existing client on new data
  teardown     stop hosting + revoke API key
  status       show state of a client engagement
"""
from __future__ import annotations

# CUDA runtime compat MUST happen before any heavy imports — re-execs the
# process with proper LD_LIBRARY_PATH if not already set.
import ops.cuda_compat   # noqa: F401

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler

ROOT = Path(__file__).parent
CONFIGS_DIR = ROOT / "configs"
CLIENTS_DIR = ROOT / "clients"
CLIENTS_DIR.mkdir(exist_ok=True)

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
log = logging.getLogger("pylox")


def load_tier_config(tier: str) -> dict:
    """Load configs/tier-{tier}-*.yml — fuzzy match on tier name."""
    matches = list(CONFIGS_DIR.glob(f"tier-{tier}-*.yml"))
    if not matches:
        raise click.ClickException(f"No config found for tier '{tier}'. Available: " +
                                   ", ".join(p.stem.replace("tier-", "") for p in CONFIGS_DIR.glob("tier-*.yml")))
    if len(matches) > 1:
        raise click.ClickException(f"Multiple configs match tier '{tier}': {[p.name for p in matches]}")
    return yaml.safe_load(matches[0].read_text())


def client_dir(client: str) -> Path:
    d = CLIENTS_DIR / client
    d.mkdir(exist_ok=True)
    return d


def write_client_state(client: str, **updates) -> dict:
    """Merge updates into clients/{client}/state.json."""
    state_path = client_dir(client) / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    state.update(updates)
    state_path.write_text(json.dumps(state, indent=2, default=str))
    return state


def read_client_state(client: str) -> dict:
    state_path = client_dir(client) / "state.json"
    return json.loads(state_path.read_text()) if state_path.exists() else {}


@click.group()
@click.version_option("0.1.0", prog_name="pylox-pipeline")
def cli() -> None:
    """Pylox Systems fine-tune pipeline."""


@cli.command()
@click.option("--client", required=True, help="Client slug (e.g. 'acme-health')")
@click.option("--tier", required=True, type=click.Choice(["8b", "32b", "70b"]))
@click.option("--data", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--target", default="runpod", type=click.Choice(["runpod", "spark"]))
@click.option("--skip-enrich", is_flag=True, help="Skip gpt-oss-120b enrichment (dev mode)")
@click.option("--dry-run", is_flag=True, help="Print plan, don't execute")
def onboard(client: str, tier: str, data: Path, target: str, skip_enrich: bool, dry_run: bool) -> None:
    """Run full pipeline for a new client: intake -> enrich -> train -> eval -> deploy."""
    config = load_tier_config(tier)
    log.info(f"[bold green]Onboarding[/bold green] {client} on tier [cyan]{tier}[/cyan] -> target [yellow]{target}[/yellow]")

    if dry_run:
        console.print("[yellow]DRY RUN — no work performed.[/yellow]")
        console.print(f"Steps: intake -> {'(skip enrich)' if skip_enrich else 'enrich'} -> train -> eval -> deploy[{target}]")
        console.print(f"Config: {config['tier']} / {config['base_model']}")
        return

    write_client_state(client, tier=tier, target=target, status="starting")

    from intake.validate import validate_data
    from intake.format_chat import format_for_chat
    validated = validate_data(data, client_dir(client))
    formatted = format_for_chat(validated, client_dir(client), config)
    write_client_state(client, intake_done=True, formatted_path=str(formatted))

    if not skip_enrich:
        from enrich.pipeline import run_enrichment
        enriched = run_enrichment(formatted, client_dir(client), config)
        write_client_state(client, enrich_done=True, enriched_path=str(enriched))
    else:
        enriched = formatted
        log.warning("Skipping enrichment — using raw formatted data")

    from train.qlora import train as train_adapter
    adapter_dir = train_adapter(enriched, client_dir(client), config, client)
    write_client_state(client, train_done=True, adapter_path=str(adapter_dir))

    from eval.runner import run_eval
    eval_report = run_eval(adapter_dir, enriched, client_dir(client), config)
    write_client_state(client, eval_done=True, eval_report=str(eval_report))

    from deploy.launcher import deploy_adapter
    endpoint = deploy_adapter(adapter_dir, client, config, target)
    write_client_state(client, deploy_done=True, endpoint=endpoint, status="live")

    console.print(f"[bold green]✓[/bold green] Onboarding complete. Endpoint: [cyan]{endpoint}[/cyan]")
    console.print(f"  Eval report: {eval_report}")


@cli.command()
@click.option("--client", required=True)
@click.option("--tier", required=True, type=click.Choice(["8b", "32b", "70b"]))
@click.option("--data", required=True, type=click.Path(exists=True, path_type=Path))
def intake(client: str, tier: str, data: Path) -> None:
    """Validate + format client data, nothing else."""
    config = load_tier_config(tier)
    from intake.validate import validate_data
    from intake.format_chat import format_for_chat
    validated = validate_data(data, client_dir(client))
    formatted = format_for_chat(validated, client_dir(client), config)
    console.print(f"[bold green]✓[/bold green] Intake complete: {formatted}")
    write_client_state(client, tier=tier, intake_done=True, formatted_path=str(formatted))


@cli.command()
@click.option("--client", required=True)
def enrich(client: str) -> None:
    """Run gpt-oss-120b enrichment on already-formatted client data."""
    state = read_client_state(client)
    if not state.get("formatted_path"):
        raise click.ClickException(f"Client {client} has no formatted data. Run `intake` first.")
    config = load_tier_config(state["tier"])
    from enrich.pipeline import run_enrichment
    enriched = run_enrichment(Path(state["formatted_path"]), client_dir(client), config)
    write_client_state(client, enrich_done=True, enriched_path=str(enriched))
    console.print(f"[bold green]✓[/bold green] Enrichment complete: {enriched}")


@cli.command()
@click.option("--client", required=True)
def train(client: str) -> None:
    """Fine-tune LoRA adapter on enriched client data."""
    state = read_client_state(client)
    data_path = Path(state.get("enriched_path") or state.get("formatted_path", ""))
    if not data_path.exists():
        raise click.ClickException(f"No training data for {client}. Run `intake` (and optionally `enrich`) first.")
    config = load_tier_config(state["tier"])
    from train.qlora import train as train_adapter
    adapter_dir = train_adapter(data_path, client_dir(client), config, client)
    write_client_state(client, train_done=True, adapter_path=str(adapter_dir))
    console.print(f"[bold green]✓[/bold green] Training complete: {adapter_dir}")


@cli.command()
@click.option("--client", required=True)
def evaluate(client: str) -> None:
    """Generate eval report for a client's trained adapter."""
    state = read_client_state(client)
    if not state.get("adapter_path"):
        raise click.ClickException(f"Client {client} has no adapter. Run `train` first.")
    config = load_tier_config(state["tier"])
    data_path = Path(state.get("enriched_path") or state.get("formatted_path"))
    from eval.runner import run_eval
    report = run_eval(Path(state["adapter_path"]), data_path, client_dir(client), config)
    write_client_state(client, eval_done=True, eval_report=str(report))
    console.print(f"[bold green]✓[/bold green] Eval report: {report}")


@cli.command()
@click.option("--client", required=True)
@click.option("--target", default="runpod", type=click.Choice(["runpod", "spark"]))
def deploy(client: str, target: str) -> None:
    """Push adapter and serve via vLLM (runpod or spark)."""
    state = read_client_state(client)
    if not state.get("adapter_path"):
        raise click.ClickException(f"Client {client} has no adapter. Run `train` first.")
    config = load_tier_config(state["tier"])
    from deploy.launcher import deploy_adapter
    endpoint = deploy_adapter(Path(state["adapter_path"]), client, config, target)
    write_client_state(client, deploy_done=True, target=target, endpoint=endpoint, status="live")
    console.print(f"[bold green]✓[/bold green] Deployed. Endpoint: [cyan]{endpoint}[/cyan]")


@cli.command()
@click.option("--client", required=True)
@click.option("--data", required=True, type=click.Path(exists=True, path_type=Path))
def refresh(client: str, data: Path) -> None:
    """Re-train existing client on new data. Reuses their tier + target."""
    state = read_client_state(client)
    if not state.get("tier"):
        raise click.ClickException(f"Client {client} not onboarded yet. Use `onboard` first.")
    target = state.get("target", "runpod")
    ctx = click.get_current_context()
    ctx.invoke(onboard, client=client, tier=state["tier"], data=data, target=target, skip_enrich=False, dry_run=False)


@cli.command()
@click.option("--client", required=True)
def teardown(client: str) -> None:
    """Stop hosting + revoke API key for a client."""
    state = read_client_state(client)
    if state.get("target") == "runpod":
        from deploy.runpod_client import stop_pod
        pod_id = state.get("runpod_pod_id")
        if pod_id:
            stop_pod(pod_id)
            log.info(f"Stopped RunPod pod {pod_id}")
    elif state.get("target") == "spark":
        from deploy.spark import stop_client_endpoint
        stop_client_endpoint(client)
    write_client_state(client, status="torn_down")
    console.print(f"[bold yellow]•[/bold yellow] {client} torn down.")


@cli.command()
@click.option("--client", required=False, help="If omitted, list all clients.")
def status(client: Optional[str]) -> None:
    """Show state of one client or all clients."""
    if client:
        state = read_client_state(client)
        if not state:
            console.print(f"[red]No state for {client}.[/red]")
            return
        console.print(f"[bold]{client}[/bold]")
        for k, v in state.items():
            console.print(f"  {k}: {v}")
    else:
        clients = [p for p in CLIENTS_DIR.iterdir() if p.is_dir()]
        if not clients:
            console.print("[yellow]No clients onboarded yet.[/yellow]")
            return
        for c in sorted(clients):
            s = read_client_state(c.name)
            console.print(f"[cyan]{c.name}[/cyan] — tier={s.get('tier','?')} status={s.get('status','?')} endpoint={s.get('endpoint','—')}")


if __name__ == "__main__":
    cli()
