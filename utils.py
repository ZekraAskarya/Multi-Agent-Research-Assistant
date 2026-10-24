def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str)
            else block.get("text", "") if isinstance(block, dict) and block.get("type") == "text"
            else ""
            for block in content
        )
    return str(content)
