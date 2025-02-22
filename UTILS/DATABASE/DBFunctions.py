import ast
import json
import uuid
from datetime import datetime
from typing import List, Tuple, Optional, Any, Dict

from loguru import logger

class DBFunctions:
    def __init__(self, DB):
        self.db = DB

    def insert(self, table: str, data: Dict[str, Any], returning: str = None) -> Optional[Any]:
        """
        Generic insert function that handles dictionary of column-value pairs.

        Args:
            table: Table name to insert into
            data: Dictionary of column names and their values
            returning: Optional column to return after insert

        Returns:
            Optional returned value if returning is specified
        """
        columns = list(data.keys())
        values = [self._format_value(v) for v in data.values()]

        insert_string = f"""
            INSERT INTO {table} ({', '.join(columns)})
            VALUES ({', '.join(values)})
        """

        if returning:
            insert_string += f" RETURNING {returning}"

        return self.db.execute(execution_string=insert_string, commit=True,
                               fetch_one=bool(returning))

    def update(self, table: str, data: Dict[str, Any],
               where: Dict[str, Any], returning: str = None) -> Optional[Any]:
        """
        Generic update function that handles dictionary of values and conditions.

        Args:
            table: Table name to update
            data: Dictionary of column names and their new values
            where: Dictionary of column names and values for WHERE clause
            returning: Optional column to return after update

        Returns:
            Optional returned value if returning is specified
        """
        set_clause = ', '.join([
            f"{k} = {self._format_value(v)}" for k, v in data.items()
        ])
        where_clause = ' AND '.join([
            f"{k} = {self._format_value(v)}" for k, v in where.items()
        ])

        update_string = f"""
            UPDATE {table}
            SET {set_clause}
            WHERE {where_clause}
        """

        if returning:
            update_string += f" RETURNING {returning}"

        return self.db.execute(execution_string=update_string, commit=True,
                               fetch_one=bool(returning))

    def select(self, table: str, columns: [str, List[str]] = None,
               where: Dict[str, Any] = None, order_by: str = None,
               limit: int = None, fetch_all: bool = True) -> List[Tuple] | Tuple:
        """
        Generic select function with support for various clauses.

        Args:
            table: Table name to select from
            columns: List of columns to select, defaults to all
            where: Dictionary of column names and values for WHERE clause
            order_by: Optional ORDER BY clause
            limit: Optional LIMIT clause
            fetch_all: Whether to fetch all results or just one

        Returns:
            List of tuples for fetch_all=True, single tuple for fetch_all=False
        """
        cols = ', '.join(columns) if columns else '*'
        select_string = f"SELECT {cols} FROM {table}"

        if where:
            where_clause = ' AND '.join([
                f"{k} = {self._format_value(v)}" for k, v in where.items()
            ])
            select_string += f" WHERE {where_clause}"

        if order_by:
            select_string += f" ORDER BY {order_by}"

        if limit:
            select_string += f" LIMIT {limit}"


        return self.db.execute(execution_string=select_string,
                               fetch_all=fetch_all,
                               fetch_one=not fetch_all)

    def delete(self, table: str, where: Dict[str, Any],
               returning: str = None) -> Optional[Any]:
        """
        Generic delete function with WHERE clause support.

        Args:
            table: Table name to delete from
            where: Dictionary of column names and values for WHERE clause
            returning: Optional column to return after delete

        Returns:
            Optional returned value if returning is specified
        """
        where_clause = ' AND '.join([
            f"{k} = {self._format_value(v)}" for k, v in where.items()
        ])

        delete_string = f"""
            DELETE FROM {table}
            WHERE {where_clause}
        """

        if returning:
            delete_string += f" RETURNING {returning}"

        return self.db.execute(execution_string=delete_string, commit=True,
                               fetch_one=bool(returning))

    @staticmethod
    def sanitize_json(data: Any) -> Any:
        """
        Recursively traverse the JSON data and replace ' and " with $ in all string values.

        Args:
            data: The JSON data (dict, list, or other types)

        Returns:
            The sanitized JSON data with ' and " replaced by $
        """
        if isinstance(data, dict):
            return {k: DBFunctions.sanitize_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [DBFunctions.sanitize_json(element) for element in data]
        elif isinstance(data, str):
            return data.replace("'", "$").replace('"', "$")
        else:
            return data

    def _format_value(self, value: Any) -> str:
        """Format the value for SQL insertion, sanitizing JSON structures."""
        if value is None:
            return 'NULL'
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, bool):
            return str(value).upper()
        elif isinstance(value, (dict, list)):
            sanitized_data = self.sanitize_json(value)
            json_str = json.dumps(sanitized_data)
            return f"'{json_str}'"
        elif isinstance(value, datetime):
            return f"'{value.isoformat()}'"
        else:
            # For non-JSON string types, retain quotes as per normal SQL syntax
            sanitized_str = str(value).replace("'", "''")  # Escape single quotes
            return f"'{sanitized_str}'"





if __name__ == '__main__':
    import os



