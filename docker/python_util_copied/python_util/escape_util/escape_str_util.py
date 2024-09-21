def escape_empty_str_quotes(in_str):
    if isinstance(in_str, str):
        return f'\"{in_str}\"'
    return in_str