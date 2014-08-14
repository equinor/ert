if (ERT_DOC)
	add_custom_command(OUTPUT doc_python
                   COMMAND sphinx-apidoc -e -o python ../python/python/
                   COMMAND make html
                   )

	add_custom_target(part_out
                  DEPENDS part.out)