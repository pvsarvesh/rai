"""
Database connection and data fetching module.
"""

import logging
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, TABLE_NAME

def get_db_engine():
    """
    Create a SQLAlchemy engine for MySQL connection.
    
    Returns:
        engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine object
    """
    try:
        engine = create_engine(
            f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
            echo=False,
            pool_pre_ping=True
        )
        logging.info("Database engine created successfully.")
        return engine
    except Exception as e:
        logging.exception("Failed to create database engine.")
        raise RuntimeError(f"Error creating database engine: {e}")

from sqlalchemy import text

def fetch_table_data(engine, table_name=TABLE_NAME):
    """
    Fetch data from the specified table.
    """
    try:
        query = text(f"SELECT * FROM `{table_name}`")  # <-- SQLAlchemy-safe SQL object
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn)  # <-- Use connection, not engine
        logging.info(f"Fetched {len(df)} records from table: {table_name}")
        return df
    except SQLAlchemyError as e:
        logging.exception("SQLAlchemy error while fetching data.")
        raise RuntimeError(f"SQLAlchemy error while fetching table '{table_name}': {e}")
    except Exception as e:
        logging.exception("General error while fetching data.")
        raise RuntimeError(f"Error fetching table '{table_name}': {e}")
