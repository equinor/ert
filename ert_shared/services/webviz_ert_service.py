from ert_shared.services._base_service import BaseService, local_exec_args


class WebvizErt(BaseService):
    service_name = "webviz-ert"

    def __init__(self):
        exec_args = local_exec_args("webviz_ert")
        super().__init__(exec_args)
