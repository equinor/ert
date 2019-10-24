def translate_(config):
    lines = []
    for group in config:
        prefix = group.name

        if group.type != 'array':
            raise NotImplementedError()

        for var in group.variables:
            name = '{}_{}'.format(prefix, var.name)

            if var.distribution == 'uniform':
                lines.append('{} UNIFORM {} {}'.format(name, var.min, var.max))
            elif var.distribution == 'const':
                lines.append('{} CONST {}'.format(name, var.value))
            else:
                raise NotImplementedError()

    return lines
