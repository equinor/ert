from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

resource = Resource(attributes={SERVICE_NAME: "everest"})
tracer_provider = TracerProvider(
    resource=resource, span_limits=SpanLimits(max_events=128 * 16)
)

tracer = trace.get_tracer("everest.main", tracer_provider=tracer_provider)


def get_trace_id() -> str:
    return trace.format_trace_id(trace.get_current_span().get_span_context().trace_id)


def get_traceparent() -> str | None:
    carrier: dict[str, str] = {}
    # Write the current context into the carrier.
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent")
