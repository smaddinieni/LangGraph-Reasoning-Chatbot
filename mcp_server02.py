import os
import pyodbc
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

conn_str = os.getenv("SQL_CONN_STR")

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("SQL_Server")



def get_db_connection() -> pyodbc.Connection:
    """
    Establishes and returns a new database connection using the connection string from environment variables.
    Returns:
        pyodbc.Connection: Active database connection.
    Raises:
        ValueError: If the connection string is not set.
    """
    if not conn_str:
        raise ValueError("Database connection string not set in environment variables.")
    return pyodbc.connect(conn_str)


def _ensure_not_chainlit_logs(table: str) -> None:
    """
    Raises a ValueError if the table is 'chainlit_logs' (case-insensitive).
    """
    if table.strip().lower() == "chainlit_logs":
        raise ValueError("Operation on 'chainlit_logs' table is not permitted.")
    

@mcp.tool()
def list_tables() -> List[str]:
    """
    Lists all user tables in the connected SQL Server database, excluding the 'chainlit_logs' table.

    Returns:
        List[str]: List of table names (excluding 'chainlit_logs').
    """
    query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
          AND LOWER(TABLE_NAME) <> 'chainlit_logs'
        ORDER BY TABLE_NAME
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        tables = [row[0] for row in cursor.fetchall()]
    return tables


@mcp.tool()
def select_from_table(
    table: str,
    columns: Optional[List[str]] = None,
    where: Optional[str] = None,
    params: Optional[List[Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Executes a SELECT query on the specified table.
    Args:
        table (str): Table name.
        columns (Optional[List[str]]): List of columns to select. Defaults to all (*).
        where (Optional[str]): WHERE clause (without 'WHERE').
        params (Optional[List[Any]]): Parameters for the WHERE clause.
        order_by (Optional[str]): ORDER BY clause (without 'ORDER BY').
        limit (Optional[int]): Limit the number of returned rows.
    Returns:
        List[Dict[str, Any]]: List of rows as dictionaries.
    """

    _ensure_not_chainlit_logs(table)

    columns_sql = ", ".join([f"[{col}]" for col in columns]) if columns else "*"
    sql = f"SELECT {columns_sql} FROM [{table}]"
    if where:
        sql += f" WHERE {where}"
    if order_by:
        sql += f" ORDER BY {order_by}"
    if limit:
        sql += f" OFFSET 0 ROWS FETCH NEXT {limit} ROWS ONLY"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or [])
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


@mcp.tool()
def insert_into_table(
    table: str,
    data: Dict[str, Any]
) -> int:
    """
    Inserts a new row into the specified table.
    Args:
        table (str): Table name.
        data (Dict[str, Any]): Dictionary of column-value pairs.
    Returns:
        int: Number of rows inserted.
    """
    _ensure_not_chainlit_logs(table)

    columns = ", ".join([f"[{col}]" for col in data.keys()])
    placeholders = ", ".join(["?" for _ in data])
    sql = f"INSERT INTO [{table}] ({columns}) VALUES ({placeholders})"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, list(data.values()))
        conn.commit()
        return cursor.rowcount


@mcp.tool()
def update_table(
    table: str,
    data: Dict[str, Any],
    where: str,
    params: Optional[List[Any]] = None
) -> int:
    """
    Updates rows in the specified table.
    Args:
        table (str): Table name.
        data (Dict[str, Any]): Dictionary of column-value pairs to update.
        where (str): WHERE clause (without 'WHERE').
        params (Optional[List[Any]]): Parameters for the WHERE clause.
    Returns:
        int: Number of rows updated.
    """
    _ensure_not_chainlit_logs(table)
    set_clause = ", ".join([f"[{col}] = ?" for col in data.keys()])
    sql = f"UPDATE [{table}] SET {set_clause} WHERE {where}"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, list(data.values()) + (params or []))
        conn.commit()
        return cursor.rowcount


@mcp.tool()
def delete_from_table(
    table: str,
    where: str,
    params: Optional[List[Any]] = None
) -> int:
    """
    Deletes rows from the specified table.
    Args:
        table (str): Table name.
        where (str): WHERE clause (without 'WHERE').
        params (Optional[List[Any]]): Parameters for the WHERE clause.
    Returns:
        int: Number of rows deleted.
    """
    _ensure_not_chainlit_logs(table)
    sql = f"DELETE FROM [{table}] WHERE {where}"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or [])
        conn.commit()
        return cursor.rowcount


@mcp.tool()
def create_table(
    table: str,
    columns: Dict[str, str]
) -> None:
    """
    Creates a new table with the specified columns.
    Args:
        table (str): Table name.
        columns (Dict[str, str]): Dictionary of column names and their SQL types.
    """
    _ensure_not_chainlit_logs(table)
    columns_sql = ", ".join([f"[{col}] {dtype}" for col, dtype in columns.items()])
    sql = f"CREATE TABLE [{table}] ({columns_sql})"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()


@mcp.tool()
def alter_table_add_column(
    table: str,
    column: str,
    dtype: str
) -> None:
    """
    Adds a new column to an existing table.
    Args:
        table (str): Table name.
        column (str): Column name.
        dtype (str): SQL data type.
    """
    _ensure_not_chainlit_logs(table)
    sql = f"ALTER TABLE [{table}] ADD [{column}] {dtype}"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()


@mcp.tool()
def drop_table(
    table: str
) -> None:
    """
    Drops the specified table.
    Args:
        table (str): Table name.
    """
    _ensure_not_chainlit_logs(table)
    sql = f"DROP TABLE [{table}]"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()


@mcp.tool()
def join_tables(
    left_table: str,
    right_table: str,
    join_condition: str,
    columns: Optional[List[str]] = None,
    join_type: str = "INNER"
) -> List[Dict[str, Any]]:
    """
    Performs a JOIN between two tables.
    Args:
        left_table (str): Left table name.
        right_table (str): Right table name.
        join_condition (str): SQL join condition (e.g., 'a.id = b.a_id').
        columns (Optional[List[str]]): Columns to select.
        join_type (str): Type of join ('INNER', 'LEFT', etc.).
    Returns:
        List[Dict[str, Any]]: List of rows as dictionaries.
    """
    _ensure_not_chainlit_logs(left_table)
    _ensure_not_chainlit_logs(right_table)
    columns_sql = ", ".join(columns) if columns else "*"
    sql = (
        f"SELECT {columns_sql} FROM [{left_table}] "
        f"{join_type} JOIN [{right_table}] ON {join_condition}"
    )
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


@mcp.tool()
def select_group_by(
    table: str,
    group_by: str,
    aggregate: str,
    where: Optional[str] = None,
    params: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    Executes a GROUP BY query with aggregation.
    Args:
        table (str): Table name.
        group_by (str): Column to group by.
        aggregate (str): Aggregate function and column (e.g., 'COUNT(*)').
        where (Optional[str]): WHERE clause (without 'WHERE').
        params (Optional[List[Any]]): Parameters for the WHERE clause.
    Returns:
        List[Dict[str, Any]]: List of rows as dictionaries.
    """
    _ensure_not_chainlit_logs(table)
    sql = f"SELECT [{group_by}], {aggregate} FROM [{table}]"
    if where:
        sql += f" WHERE {where}"
    sql += f" GROUP BY [{group_by}]"
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params or [])
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
# ...existing code...

if __name__ == "__main__":
    logger.info("Starting simplified MCP Server...")
    mcp.run(transport="stdio")