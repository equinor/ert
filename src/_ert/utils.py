def file_safe_timestamp(timestamp: str) -> str:
    """
    Convert an ISO timestamp string to a file-safe version

    Keeps the date in the extended format (YYYY-MM-DD) and converts the time
    to the basic format (HHMMSS) by removing colons. This mix is not strictly
    ISO 8601 compliant, but it can still be parsed.

    Example:
    2025-10-10T14:30:00 -> 2025-10-10T143000
    """
    return str(timestamp).replace(":", "")
