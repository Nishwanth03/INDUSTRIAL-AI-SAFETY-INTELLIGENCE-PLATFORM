"""Worker-facing safety reminders for validated incident types."""

WORKER_GUIDANCE = {
    "helmet_missing": (
        "Pause before entering the monitored work area. Put on the site-approved "
        "safety helmet and check its fit according to site procedure. If it is "
        "unavailable or damaged, stay in the designated safe area and contact "
        "your supervisor."
    ),
    "vest_missing": (
        "Pause before entering the monitored work area. Wear the required "
        "site-approved reflective vest. If it is unavailable or damaged, stay "
        "in the designated safe area and contact your supervisor."
    ),
    "shoes_missing": (
        "Do not enter the hazardous work area without the required site-approved "
        "safety footwear. If it is unavailable or damaged, stay in the designated "
        "safe area and contact your supervisor."
    ),
    "restricted_entry": (
        "Do not enter the restricted area. If you are already inside, leave by "
        "the designated safe route when safe to do so and notify your supervisor."
    ),
    "uncertain": (
        "PPE status could not be reliably confirmed. Do not treat this as approval "
        "to proceed; ask a trained supervisor to verify and follow site procedure."
    ),
}


def get_worker_guidance(issue_type: str) -> str:
    """Return a safety reminder for a known issue type."""
    try:
        return WORKER_GUIDANCE[issue_type]
    except KeyError as exc:
        raise ValueError(f"Unknown safety issue type: {issue_type}") from exc
