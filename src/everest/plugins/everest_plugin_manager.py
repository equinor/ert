from typing import Any, Dict, List, Tuple, Type, TypeVar, Union
import pluggy

from ert.config.forward_model_step import ForwardModelStepDocumentation, ForwardModelStepPlugin
from everest.plugins import hook_impl, hook_specs
from everest.strings import EVEREST
import logging
from .plugin_response import PluginMetadata, PluginResponse

logger = logging.getLogger(__name__)




K = TypeVar("K")
V = TypeVar("V")


class EverestPluginManager(pluggy.PluginManager):
    def __init__(self, plugins=None):
        super(EverestPluginManager, self).__init__(EVEREST)
        self.add_hookspecs(hook_specs)
        if plugins is None:
            self.register(hook_impl)
            self.load_setuptools_entrypoints(EVEREST)
        else:
            for plugin in plugins:
                self.register(plugin)

    @property
    def forward_model_steps(
        self,
    ) -> List[Type[ForwardModelStepPlugin]]:
        fm_steps_listed = [
            resp.data for resp in self.hook.installable_forward_model_steps()
        ]
        return [fm_step for fm_steps in fm_steps_listed for fm_step in fm_steps]


    @staticmethod
    def _add_plugin_info_to_dict(
        d: Dict[K, V], plugin_response: PluginResponse[Any]
    ) -> Dict[K, Tuple[V, PluginMetadata]]:
        return {k: (v, plugin_response.plugin_metadata) for k, v in d.items()}

    @staticmethod
    def _merge_dicts(
        list_of_dicts: List[PluginResponse[Dict[str, V]]],
        include_plugin_data: bool = False,
    ) -> Union[Dict[str, V], Dict[str, Tuple[V, PluginMetadata]]]:
        list_of_dicts.reverse()
        merged_dict: Dict[str, Tuple[V, PluginMetadata]] = {}
        for d in list_of_dicts:
            conflicting_keys = set(merged_dict.keys()) & set(d.data.keys())
            for ck in conflicting_keys:
                logger.info(
                    f"Overwriting {ck} from "
                    f"{merged_dict[ck][1].plugin_name}"
                    f"({merged_dict[ck][1].function_name}) "
                    f"with data from {d.plugin_metadata.plugin_name}"
                    f"({d.plugin_metadata.function_name})"
                )
            merged_dict.update(EverestPluginManager._add_plugin_info_to_dict(d.data, d))
        if include_plugin_data:
            return merged_dict
        return {k: v[0] for k, v in merged_dict.items()}            

    @staticmethod
    def _evaluate_job_doc_hook(
        hook: pluggy.HookCaller, job_name: str
    ) -> Dict[Any, Any]:
        response = hook(job_name=job_name)

        if response is None:
            logger.debug(f"Got no documentation for {job_name} from any plugins")
            return {}

        return response.data



    def get_documentation_for_jobs(self) -> Dict[str, Any]:
        print(self.hook.get_forward_models())
        job_docs = {
            k: {
                "config_file": v[0],
                "source_package": v[1].plugin_name,
                "source_function_name": v[1].function_name,
            }
            
            for k, v in EverestPluginManager._merge_dicts(
                self.hook.installable_jobs(), include_plugin_data=True
            ).items()
        }
        for key, value in job_docs.items():
            value.update(
                EverestPluginManager._evaluate_job_doc_hook(
                    self.hook.job_documentation,
                    key,
                )
            )
        return job_docs

    def get_documentation_for_forward_model_steps(
        self,
    ) -> Dict[str, ForwardModelStepDocumentation]:
        return {
            # Implementations of plugin fm step take no __init__ args
            # (name, command)
            # but mypy expects the subclasses to take in same arguments upon
            # initializations
            fm_step().name: fm_step.documentation()  # type: ignore
            for fm_step in self.forward_model_steps
            if fm_step.documentation() is not None
        }

