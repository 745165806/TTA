import csv
from pathlib import Path

from eptta.data.adapters.base import protocol_context
from eptta.data.contracts import RawRecord
from eptta.errors import ContractError, DataError


class DelimitedAdapter:
    """Parse headered/headerless CSV, TSV, or whitespace under an explicit contract."""
    def iter_file(self, path, payload):
        source = Path(path)
        encoding = payload["encoding"]
        delimiter = payload["delimiter"]
        columns = payload["columns"]
        header = payload["header"]
        context = protocol_context(source, payload.get("protocol_contexts"))
        if not isinstance(columns, dict) or not columns:
            raise ContractError("explicit logical-to-source columns are required")
        with source.open(encoding=encoding, newline="") as stream:
            if delimiter == "whitespace":
                rows = ((row, line.split()) for row, line in enumerate(stream, 1) if line.strip())
                names = None
                if header:
                    try:
                        _, names = next(rows)
                    except StopIteration:
                        raise DataError(f"empty protocol: {source}")
            else:
                if len(delimiter) != 1:
                    raise ContractError("delimiter must be one character or 'whitespace'")
                reader = csv.reader(stream, delimiter=delimiter)
                rows = enumerate(reader, 1)
                names = None
                if header:
                    try:
                        _, names = next(rows)
                    except StopIteration:
                        raise DataError(f"empty protocol: {source}")
            for row_number, row in rows:
                try:
                    fields = {}
                    for logical, source_column in columns.items():
                        if header:
                            if not isinstance(source_column, str) or source_column not in names:
                                raise DataError(f"reviewed header column absent: {source_column!r}")
                            fields[logical] = row[names.index(source_column)]
                        else:
                            if type(source_column) is not int:
                                raise ContractError("headerless columns must use integer indices")
                            fields[logical] = row[source_column]
                except IndexError as exc:
                    raise DataError(f"short row at {source}:{row_number}") from exc
                yield RawRecord(f"{source}:{row_number}", fields, str(source), row_number, context)
