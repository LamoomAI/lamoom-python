def get_common_args(max_tokens):
    return {
        "top_p": 1,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
