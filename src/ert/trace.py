from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider

resource = Resource(attributes={SERVICE_NAME: "ert"})
tracer_provider = TracerProvider(
    resource=resource, span_limits=SpanLimits(max_events=128 * 16)
)
trace.set_tracer_provider(tracer_provider)

tracer = trace.get_tracer("ert.main")


def get_trace_id() -> str:
    return trace.format_trace_id(trace.get_current_span().get_span_context().trace_id)
