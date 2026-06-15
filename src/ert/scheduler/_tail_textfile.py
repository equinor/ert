from pathlib import Path


def tail_textfile(file_path: Path, num_chars: int) -> str:
    if not file_path.exists():
        return f"No output file {file_path}"
    with file_path.open(encoding="utf-8") as file:
        file.seek(0, 2)
        file_end_position = file.tell()
        seek_position = max(0, file_end_position - num_chars)
        file.seek(seek_position)
        return file.read()[-num_chars:]
