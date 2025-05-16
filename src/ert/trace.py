from opentelemetry import trace
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, SpanProcessor, TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

ThreadingInstrumentor().instrument()

resource = Resource(attributes={SERVICE_NAME: "ert"})
tracer_provider = TracerProvider(
    resource=resource, span_limits=SpanLimits(max_events=128 * 16)
)
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer("ert.trace")


def get_trace_id() -> str:
    return trace.format_trace_id(trace.get_current_span().get_span_context().trace_id)


def get_traceparent() -> str | None:
    carrier: dict[str, str] = {}
    # Write the current context into the carrier.
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent")


def add_span_processor(span_processor: SpanProcessor) -> None:
    # We don't want the same span processor registered multiple times,
    # since this will cause duplicate span entries.
    if span_processor not in tracer_provider._active_span_processor._span_processors:
        tracer_provider.add_span_processor(span_processor)
