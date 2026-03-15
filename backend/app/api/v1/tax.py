"""Tax reporting endpoints."""
import csv
import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies import require_auth
from app.models.base import async_session_factory

router = APIRouter(prefix="/tax-report", tags=["tax"])


@router.get("/{year}")
async def get_tax_report(year: int, _user: dict = Depends(require_auth)):
    from app.services.reporting.tax import tax_reporter
    async with async_session_factory() as db:
        report = await tax_reporter.generate_annual_report(db, year)
        return {
            "year": report.year,
            "total_realized_gains": report.total_realized_gains,
            "total_realized_losses": report.total_realized_losses,
            "net_realized": report.net_realized,
            "total_trades": report.total_trades,
            "total_fees": report.total_fees,
            "net_after_fees": report.net_after_fees,
            "by_month": report.trades_by_month,
        }


@router.get("/{year}/csv")
async def get_tax_csv(year: int, _user: dict = Depends(require_auth)):
    """Download tax report as CSV."""
    from app.services.reporting.tax import tax_reporter

    async with async_session_factory() as db:
        rows = await tax_reporter.generate_csv_data(db, year)

    if not rows:
        raise HTTPException(status_code=404, detail="No trades found for this year")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=tax_report_{year}.csv"},
    )
