# -*- coding: utf-8 -*-
"""stock-db.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZKlL86mvAxk0cN3YMYYQZoYwkCViKi1c
"""

import pandas as pd
import sqlite3

conn = sqlite3.connect("stocks.sqlite")
conn

"""- Get the table & columns in data"""

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables

columns = pd.read_sql("SELECT * FROM STOCK_DATA", conn)
columns

"""# Create two new tables in the database (stocks.sqlite), one with only 'MSFT' values for the Symbol feature, and one with only 'AAPL' values for the Symbol feature."""

conn.execute("CREATE TABLE IF NOT EXISTS MSFT_table AS SELECT * FROM STOCK_DATA WHERE Symbol = 'MSFT'")
conn.execute("CREATE TABLE IF NOT EXISTS AAPL_table AS SELECT * FROM STOCK_DATA WHERE Symbol = 'AAPL'")

"""# Read the two new tables in from the database using SQL to check if they were successfully created."""

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
tables

pd.read_sql("SELECT * FROM MSFT_table", conn)

pd.read_sql("SELECT * FROM AAPL_table", conn)

"""# For each new table in the database, query for rows containing the Maximum and Minimum dates, and save those rows as new pandas data frames (2 rows per dataframe).


"""

date_MSFT = pd.read_sql(""" SELECT *
                        FROM MSFT_table
                        WHERE Date = (SELECT max(Date) FROM MSFT_table)
                        OR Date = (SELECT min(Date) FROM MSFT_table)
                        """, conn)
date_MSFT

date_AAPL = pd.read_sql(""" SELECT *
                       FROM AAPL_table
                       WHERE Date = (SELECT max(Date) FROM AAPL_table)
                       OR Date = (SELECT min(Date) FROM AAPL_table)
                       """, conn)
date_AAPL

"""# For each new table in the database, query for values greater than 50 in the Open feature, and save those as new pandas data frames."""

open_MSFT= pd.read_sql(""" SELECT *
                       FROM MSFT_table
                       WHERE Open > 50
                       """, conn)
open_MSFT

open_AAPL= pd.read_sql(""" SELECT *
                       FROM AAPL_table
                       WHERE Open > 50
                       """, conn)
open_AAPL