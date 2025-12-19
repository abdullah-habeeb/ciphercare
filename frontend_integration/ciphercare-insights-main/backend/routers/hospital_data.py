"""
Hospital Data Upload Router

Handles data ingestion for hospitals (CSV, JSON, form-based).
"""

import csv
import io
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from backend.database.database import get_session
from backend.database.models import HospitalData

router = APIRouter()


class DataUploadRequest(BaseModel):
    """JSON-based data upload request."""
    data: List[Dict[str, Any]] = Field(..., description="Array of data records")
    label: Optional[int] = Field(None, description="Optional label for supervised learning")
    source: str = Field(default="upload", description="Data source: upload, batch, or synthetic")


class DataUploadResponse(BaseModel):
    """Response after data upload."""
    message: str
    hospital_id: str
    records_inserted: int
    timestamp: datetime


@router.post("/hospitals/{hospital_id}/upload-data", response_model=DataUploadResponse)
async def upload_hospital_data(
    hospital_id: str,
    file: Optional[UploadFile] = File(None),
    json_data: Optional[str] = Form(None),
    label: Optional[int] = Form(None),
    source: str = Form("upload"),
    session: Session = Depends(get_session)
):
    """
    Upload hospital data via CSV file, JSON form data, or direct JSON.
    
    Supports:
    - CSV file upload
    - Form-based JSON input
    - Direct JSON POST
    """
    records_inserted = 0
    
    try:
        # Handle CSV file upload
        if file and file.filename:
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="Only CSV files are supported")
            
            content = await file.read()
            csv_content = content.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            records = []
            for row in csv_reader:
                # Convert CSV row to dict, handling numeric conversions
                processed_row = {}
                for key, value in row.items():
                    # Try to convert to number if possible
                    try:
                        if '.' in value:
                            processed_row[key] = float(value)
                        else:
                            processed_row[key] = int(value)
                    except (ValueError, TypeError):
                        processed_row[key] = value
                
                records.append(processed_row)
            
            # Insert records
            for record in records:
                hospital_data = HospitalData(
                    hospital_id=hospital_id,
                    raw_data=record,
                    label=label,
                    source=source
                )
                session.add(hospital_data)
                records_inserted += 1
            
            session.commit()
            
            return DataUploadResponse(
                message="CSV data uploaded successfully",
                hospital_id=hospital_id,
                records_inserted=records_inserted,
                timestamp=datetime.utcnow()
            )
        
        # Handle JSON form data
        elif json_data:
            import json
            try:
                data = json.loads(json_data)
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
            
            # Insert records
            for record in records:
                hospital_data = HospitalData(
                    hospital_id=hospital_id,
                    raw_data=record,
                    label=label,
                    source=source
                )
                session.add(hospital_data)
                records_inserted += 1
            
            session.commit()
            
            return DataUploadResponse(
                message="JSON data uploaded successfully",
                hospital_id=hospital_id,
                records_inserted=records_inserted,
                timestamp=datetime.utcnow()
            )
        
        else:
            raise HTTPException(status_code=400, detail="Either file or json_data must be provided")
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")


@router.post("/hospitals/{hospital_id}/upload-data-json", response_model=DataUploadResponse)
async def upload_hospital_data_json(
    hospital_id: str,
    request: DataUploadRequest,
    session: Session = Depends(get_session)
):
    """
    Upload hospital data via direct JSON POST.
    """
    records_inserted = 0
    
    try:
        for record in request.data:
            hospital_data = HospitalData(
                hospital_id=hospital_id,
                raw_data=record,
                label=request.label,
                source=request.source
            )
            session.add(hospital_data)
            records_inserted += 1
        
        session.commit()
        
        return DataUploadResponse(
            message="Data uploaded successfully",
            hospital_id=hospital_id,
            records_inserted=records_inserted,
            timestamp=datetime.utcnow()
        )
    
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Error uploading data: {str(e)}")


@router.get("/hospitals/{hospital_id}/data")
async def get_hospital_data(
    hospital_id: str,
    limit: int = 100,
    offset: int = 0,
    session: Session = Depends(get_session)
):
    """Get hospital data records."""
    statement = select(HospitalData).where(
        HospitalData.hospital_id == hospital_id
    ).offset(offset).limit(limit).order_by(HospitalData.timestamp.desc())
    
    results = session.exec(statement).all()
    
    return {
        "hospital_id": hospital_id,
        "count": len(results),
        "data": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "raw_data": r.raw_data,
                "label": r.label,
                "source": r.source
            }
            for r in results
        ]
    }


@router.get("/hospitals/{hospital_id}/data/count")
async def get_hospital_data_count(
    hospital_id: str,
    session: Session = Depends(get_session)
):
    """Get count of data records for a hospital."""
    statement = select(HospitalData).where(HospitalData.hospital_id == hospital_id)
    results = session.exec(statement).all()
    
    return {
        "hospital_id": hospital_id,
        "count": len(list(results))
    }

