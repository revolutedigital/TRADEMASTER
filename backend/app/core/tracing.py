"""OpenTelemetry distributed tracing setup."""
from app.core.logging import get_logger

logger = get_logger(__name__)

# Tracing is optional - gracefully degrade if not installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

    _provider = TracerProvider()
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(_provider)
    tracer = trace.get_tracer("trademaster")
    TRACING_AVAILABLE = True
    logger.info("opentelemetry_initialized")
except ImportError:
    TRACING_AVAILABLE = False
    tracer = None
    logger.info("opentelemetry_not_installed", hint="pip install opentelemetry-api opentelemetry-sdk")


def setup_tracing(app):
    """Instrument FastAPI app with OpenTelemetry (if available)."""
    if not TRACING_AVAILABLE:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("fastapi_tracing_enabled")
    except ImportError:
        logger.info("fastapi_instrumentation_not_available")
