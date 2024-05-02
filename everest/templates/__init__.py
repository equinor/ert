import pkg_resources

template_names = (
    "well_drill.tmpl",
    "well_order.tmpl",
    "schmerge.tmpl",
)


def fetch_template(template_name):
    return pkg_resources.resource_filename("everest.templates", template_name)
