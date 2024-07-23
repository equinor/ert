import ert


@ert.plugin(name="ert")  # type: ignore
def help_links():
    return {"GitHub page": "https://github.com/equinor/ert"}
