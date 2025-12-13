from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./fl_hospital.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class HospitalUpload(Base):
    __tablename__ = "hospital_uploads"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(String, index=True)
    filename = Column(String)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="processed")
    metadata_info = Column(JSON, nullable=True)

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    input_data = Column(JSON)
    prediction = Column(Float)
    risk_level = Column(String)

def init_db():
    Base.metadata.create_all(bind=engine)
