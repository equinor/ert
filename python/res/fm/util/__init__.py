from .template_render import TemplateRender

def render_template(input_files, template_file, output_file):
    TemplateRender.render_template(input_files, template_file, output_file)
