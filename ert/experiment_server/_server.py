import asyncio
import logging
import pickle
from contextlib import contextmanager
from typing import Any, Callable, Generator, Set

from cloudevents.exceptions import DataUnmarshallerError
from cloudevents.http import from_json
from websockets.legacy.server import WebSocketServerProtocol
from websockets.server import serve
from ert.serialization import evaluator_unmarshaller
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from res.enkf.enkf_main import EnKFMain

from ._experiment_protocol import Experiment
from ._registry import _Registry


logger = logging.getLogger(__name__)
event_logger = logging.getLogger("ert.event_log")


class ExperimentServer:
    """:class:`ExperimentServer` implements the experiment server API, allowing
    creation, management and running of experiments as defined by the
    :class:`ert.experiment_server._experiment_protocol.Experiment` protocol.

    :class:`ExperimentServer` also runs a server to which clients and remote
    workers can connect.
    """

    def __init__(self, ee_config: EvaluatorServerConfig) -> None:
        self._config = ee_config
        self._registry = _Registry()
        self._clients: Set[WebSocketServerProtocol] = set()
        self._server_done = asyncio.get_running_loop().create_future()
        self._server_task = asyncio.create_task(self._server())

    async def _handler(self, websocket: WebSocketServerProtocol, path: str) -> None:
        elements = path.split("/")
        if elements[1] == "client":
            await self.handle_client(websocket, path)
        elif elements[1] == "dispatch":
            logger.debug("dispatcher connected")
            await self.handle_dispatch(websocket, path)
        else:
            logger.info(f"Connection attempt to unknown path: {path}.")

    async def stop(self) -> None:
        """Stop the server."""
        logger.debug("stopping experiment server gracefully...")
        try:
            self._server_done.set_result(None)
        except asyncio.InvalidStateError:
            logger.debug("was already gracefully asked to stop.")
            pass
        await self._server_task

    async def handle_dispatch(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        """handle_dispatch(self, websocket, path: str)

        Handle incoming "dispatch" connections, which refers to remote workers.

        websocket is a https://websockets.readthedocs.io/en/stable/reference/server.html#websockets.server.WebSocketServerProtocol  # pylint: disable=line-too-long
        """
        async for msg in websocket:
            try:
                event = from_json(msg, data_unmarshaller=evaluator_unmarshaller)
            except DataUnmarshallerError:
                event = from_json(msg, data_unmarshaller=pickle.loads)

            event_logger.debug("handle_dispatch: %s", event)

            await self._registry.all_experiments[0].dispatch(event, 0)

    @contextmanager
    def store_client(
        self, websocket: WebSocketServerProtocol
    ) -> Generator[None, None, None]:
        """store_client(self, websocket)

        Context manager for a client connection handler, allowing to know how
        many clients are connected."""
        logger.debug("client %s connected", websocket)
        self._clients.add(websocket)
        yield
        self._clients.remove(websocket)

    # pylint: disable=line-too-long
    async def handle_client(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        """handle_client(self, websocket, path: str)

        Handle incoming client connections. websocket is a https://websockets.readthedocs.io/en/stable/reference/server.html#websockets.server.WebSocketServerProtocol  # pylint: disable=line-too-long
        """
        with self.store_client(websocket):
            async for message in websocket:
                client_event = from_json(
                    message, data_unmarshaller=evaluator_unmarshaller
                )
                logger.debug(f"got message from client: {client_event}")

    async def _server(self) -> None:
        try:
            async with serve(
                self._handler,
                sock=self._config.get_socket(),
                ssl=self._config.get_server_ssl_context(),
            ):
                logger.debug("Running experiment server")
                await self._server_done
            logger.debug("Async server exiting.")
        except Exception:  # pylint: disable=broad-except
            logger.exception("crash/burn")

    # pylint: disable=line-too-long
    def add_legacy_experiment(  # pylint: disable=too-many-arguments
        self,
        ert: EnKFMain,
        ensemble_size: int,
        current_case_name: str,
        args: Any,
        factory: Callable[[EnKFMain, int, str, Any], Experiment],
    ) -> str:
        """add_legacy_experiment(self, ert, ensemble_size: int,current_case_name: str, factory: Callable[[Any, int, str, Any], Experiment])

        The ert parameter, as well as the first input parameter in the factory
        :class:`Callable`, refers to the EnkfMain type.

        Create a legacy experiment using a model factory. See ``ert_shared.cli.model_factory.create_model``.
        """
        experiment = factory(
            ert,
            ensemble_size,
            current_case_name,
            args,
        )
        experiment.id_ = self._registry.add_experiment(experiment)
        return experiment.id_

    def add_experiment(self, experiment: Experiment) -> str:
        experiment.id_ = self._registry.add_experiment(experiment)
        return experiment.id_

    async def run_experiment(self, experiment_id: str) -> None:
        """Run the experiment with the given experiment_id.

        This is a helper method for use by the CLI, where only one experiment
        at a time makes sense. This method therefore runs the experiment, and
        attempts to gracefully shut down the server when complete."""
        logger.debug("running experiment %s", experiment_id)
        experiment = self._registry.get_experiment(experiment_id)

        experiment_task = asyncio.create_task(experiment.run(self._config))

        done, pending = await asyncio.wait(
            [self._server_task, experiment_task], return_when=asyncio.FIRST_COMPLETED
        )

        if experiment_task in done:
            logger.debug("experiment %s was done", experiment_id)
            # raise experiment exception if any
            try:
                experiment_task.result()
                successful_reals = await experiment.successful_realizations(0)
                # This is currently API
                print(f"Successful realizations: {successful_reals}")
            except Exception as e:  # pylint: disable=broad-except
                print(f"Experiment failed: {str(e)}")
                raise
            finally:
                # wait for shutdown of server
                await self.stop()
            return

        # experiment is pending, but the server died, so try cancelling the experiment
        # then raise the server's exception
        for p in pending:
            logger.debug("task %s was pending, cancelling...", p)
            p.cancel()
        for d in done:
            d.result()
