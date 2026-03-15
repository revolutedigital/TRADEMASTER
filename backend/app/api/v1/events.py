"""Event store and plugin system endpoints."""
from fastapi import APIRouter, Depends, Query
from app.dependencies import require_auth

router = APIRouter()


@router.get("/events")
async def list_events(
    aggregate_id: str | None = None,
    event_type: str | None = None,
    limit: int = Query(default=50, le=500),
    _user: dict = Depends(require_auth),
):
    """Query domain events from the event store."""
    from app.core.event_store import event_store
    events = event_store.get_events(
        aggregate_id=aggregate_id,
        event_type=event_type,
        limit=limit,
    )
    return {
        "events": [e.to_dict() for e in events],
        "total": len(events),
    }


@router.get("/events/stats")
async def event_store_stats(_user: dict = Depends(require_auth)):
    """Get event store statistics."""
    from app.core.event_store import event_store
    return event_store.get_stats()


@router.get("/plugins")
async def list_plugins(_user: dict = Depends(require_auth)):
    """List all registered plugins."""
    from app.core.plugin_system import plugin_manager
    return {
        "plugins": plugin_manager.get_status(),
        "total": plugin_manager.get_status()["loaded"],
    }


@router.get("/execution-analytics")
async def get_execution_analytics(_user: dict = Depends(require_auth)):
    """Get execution quality analytics."""
    from app.services.exchange.execution_analytics import execution_analytics

    report = execution_analytics.get_best_execution_report()

    return report
