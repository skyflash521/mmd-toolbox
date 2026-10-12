def format_warning_line(code, message, *, section=None):
    where = f"({_join_section(section)})" if section else ""
    prefix = f"warning: {code}: " if code else "warning: "
    return f"{prefix}{message}{where}"


def _join_section(section):
    if isinstance(section, str):
        return section
    return ",".join(section)
