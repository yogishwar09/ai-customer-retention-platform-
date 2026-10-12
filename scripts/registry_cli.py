"""Model registry operations.

    python scripts/registry_cli.py list
    python scripts/registry_cli.py stage <version>
    python scripts/registry_cli.py promote <version> <approval reason>   # also the rollback path
    python scripts/registry_cli.py archive <version>
"""

from __future__ import annotations

import sys

from retentioniq.models import registry


def main(argv: list[str]) -> int:
    if not argv or argv[0] == "list":
        reg = registry.read_registry()
        for m in sorted(reg["models"], key=lambda m: m["registered_at"], reverse=True):
            flag = "*" if m["version"] == reg.get("production") else " "
            print(f"{flag} {m['version']:<40} {m['status']:<11} PR-AUC {m['test_pr_auc']:.3f} "
                  f"ROC-AUC {m['test_roc_auc']:.3f} gate={'pass' if m['gate_passed'] else 'FAIL'}")
        return 0
    if argv[0] in ("promote", "stage", "archive") and len(argv) >= 2:
        status = {"promote": "production", "stage": "staging", "archive": "archived"}[argv[0]]
        reason = " ".join(argv[2:]) or f"{argv[0]} via CLI"
        registry.set_status(argv[1], status, "cli", reason)
        print(f"{argv[1]} -> {status}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
