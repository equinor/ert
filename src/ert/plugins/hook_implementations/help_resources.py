import ert


@ert.plugin(name="ert")
def help_links() -> dict[str, str]:
    return {"GitHub page": "https://github.com/equinor/ert"}
