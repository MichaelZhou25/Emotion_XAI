import re


def parse_sessions(value):
    """Normalize a session selector to an ordered list of session numbers."""
    if value is None:
        sessions = [1]
    elif isinstance(value, int):
        sessions = [1, 2, 3] if value == 123 else [value]
    elif isinstance(value, str):
        compact = value.strip().lower().replace('session', '').replace('s', '')
        if compact in {'123', 'all'}:
            sessions = [1, 2, 3]
        else:
            sessions = [int(x) for x in re.findall(r'\d+', compact)]
    else:
        sessions = [int(x) for x in value]

    sessions = list(dict.fromkeys(sessions))
    if not sessions or any(session not in {1, 2, 3} for session in sessions):
        raise ValueError(f'Sessions must be selected from 1, 2, and 3; got {value!r}')
    return sessions


def session_tag(sessions):
    return ''.join(str(session) for session in parse_sessions(sessions))
