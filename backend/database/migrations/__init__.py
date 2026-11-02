"""
Schema migrations, one file per change: mNNNN_short_name.py with the next free
number and an `upgrade(conn)` function.

Never edit a migration that has shipped: add a new one. Inside upgrade(), don't
call conn.commit() or conn.executescript(): either would end the transaction that
makes the migration all-or-nothing.
"""
