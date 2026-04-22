"""Render the benchmark report.md from results.json using the Jinja2 template."""
from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Template

log = logging.getLogger(__name__)


def render_report(results: dict, out_dir: Path) -> Path:
    """Flatten results dict + render markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl_path = Path(__file__).parent / "report_template.md.j2"
    tpl = Template(tpl_path.read_text())

    # Flatten top-level headline metrics for easy template access
    perf = results.get("performance") or {}
    cost = results.get("cost") or {}
    judge = results.get("llm_judge") or {}
    academic = results.get("academic") or {}

    ctx = {
        "client": results["client"],
        "tier": results["tier"],
        "base_model": results["base_model"],
        "adapter_path": results["adapter_path"],
        "endpoint": results["endpoint"],
        "timestamp": results["timestamp"],
        "perf_throughput": (perf.get("throughput_tok_s") or {}).get("single_mean"),
        "perf_ttft_p50": (perf.get("ttft_ms") or {}).get("p50"),
        "perf_ttft_p95": (perf.get("ttft_ms") or {}).get("p95"),
        "judge_finetune_pct": round(judge.get("finetune_win_rate", 0) * 100, 1) if judge else None,
        "self_hosted_cost": cost.get("self_hosted_per_1m_usd"),
        "openai_cost": 2.50,
        "cost_savings_factor": next(
            (p["vs_self_hosted_factor"] for p in cost.get("cloud_comparison", [])
             if "openai_gpt-4o" == p.get("provider")),
            None,
        ) if cost else None,
        "academic": academic,
        "domain": results.get("domain"),
        "llm_judge": judge,
        "performance": perf,
        "cost": cost,
        "extra": results.get("extra", []),
    }

    rendered = tpl.render(**ctx)
    out_path = out_dir / "report.md"
    out_path.write_text(rendered)

    # Also write a plain-text copy suitable for emailing
    txt_path = out_dir / "report.txt"
    try:
        import re
        plain = re.sub(r"[|*#_`>]", "", rendered)
        plain = re.sub(r"\n{3,}", "\n\n", plain)
        txt_path.write_text(plain)
    except Exception:
        pass

    log.info(f"Report rendered: {out_path}")
    return out_path
