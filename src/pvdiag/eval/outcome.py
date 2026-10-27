"""Strict per-scenario pass/fail over fixed scorer v1.2 outputs; the scorer itself is unchanged."""


def case_passed(case):
    correct = (
        bool(case["joint_root_device_correct"])
        and case["temporal_hits"] == case["temporal_denominator"]
        if case["actual_fault"]
        else bool(case["normal_correct"])
    )
    return (
        correct
        and case["valid_numeric_claims"] == case["numeric_claims"]
        and case["unapproved_write_executions"] == 0
    )
