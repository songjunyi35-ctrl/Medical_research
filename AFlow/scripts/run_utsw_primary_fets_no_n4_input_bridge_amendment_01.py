"""Run the frozen UTSW input bridge with fixed external-ID acceptance only."""
from __future__ import annotations

from sammed3d import t1ce_binary
from scripts import audit_utsw_primary_fets_no_n4_input_bridge as audit


ORIGINAL_CASE_ID = t1ce_binary._case_id


def external_case_id(value: object) -> str:
    if isinstance(value, str) and value in audit.SUBJECTS:
        return value
    return ORIGINAL_CASE_ID(value)


def main() -> None:
    t1ce_binary._case_id = external_case_id
    audit.main()


if __name__ == "__main__":
    main()
