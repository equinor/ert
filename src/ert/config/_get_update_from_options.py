



def get_update_from_options(options: dict[str, str], default: str | None = None) -> str | None:
    update:str | None= options.get("UPDATE", default)
    if update is None:
        return None
    if update.upper() =="NONE":
        return None
    elif update.upper() =="TRUE":
        return "ADAPTIVE"
    elif update.upper() =="FALSE":
        return None
    return update.upper()
