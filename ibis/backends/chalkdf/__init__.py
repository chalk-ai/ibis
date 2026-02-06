from __future__ import annotations

import contextlib
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pyarrow as pa

import ibis.backends.sql.compilers as sc
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends import NoUrl
from ibis.backends.sql import SQLBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd


def _import_libchalk():
    """Import libchalk from the chalkdf package.

    The native extension lives inside the chalkdf package directory,
    so we ensure that directory is on sys.path before importing.
    """
    import importlib

    try:
        return importlib.import_module("libchalk")
    except ModuleNotFoundError:
        import pathlib

        chalkdf_spec = importlib.util.find_spec("chalkdf")
        if chalkdf_spec is None or chalkdf_spec.submodule_search_locations is None:
            raise
        for loc in chalkdf_spec.submodule_search_locations:
            p = str(pathlib.Path(loc))
            if p not in sys.path:
                sys.path.insert(0, p)
        return importlib.import_module("libchalk")


class Backend(SQLBackend, NoUrl):
    name = "chalkdf"
    compiler = sc.duckdb.compiler

    def do_connect(self) -> None:
        """Initialize the chalkdf backend with an empty table registry."""
        self._tables: dict[str, pa.Table] = {}

        libchalk = _import_libchalk()
        self._chalksql = libchalk.chalksql
        self._chalktable = libchalk.chalktable
        self._chalk_utils = libchalk.utils
        self._chalk_metrics = libchalk.metrics

        chalkfunction = libchalk.chalkfunction
        function_registry = (
            chalkfunction.BASE_FUNCTIONS | chalkfunction.AGGREGATE_FUNCTIONS
        ).shadowed_by(chalkfunction.CHALK_SQL_FUNCTIONS)

        self._catalog = self._chalksql.ChalkSqlCatalog(
            function_registry=function_registry
        )
        self._env = "default"

    @property
    def version(self) -> str:
        libchalk = _import_libchalk()
        return libchalk.__version__

    def list_tables(
        self, like: str | None = None, database: tuple[str, str] | str | None = None
    ) -> list[str]:
        return self._filter_with_like(self._tables.keys(), like)

    def get_schema(
        self,
        name: str,
        /,
        *,
        catalog: str | None = None,
        database: str | None = None,
    ) -> sch.Schema:
        from ibis.formats.pyarrow import PyArrowSchema

        table = self._tables[name]
        return PyArrowSchema.to_ibis(table.schema)

    def create_table(
        self,
        name: str,
        /,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: sch.Schema | None = None,
        database: str | None = None,
        temp: bool = False,
        overwrite: bool = False,
    ) -> ir.Table:
        if isinstance(obj, pa.Table):
            self._tables[name] = obj
        elif isinstance(obj, ir.Table):
            self._tables[name] = obj.to_pyarrow()
        elif obj is not None:
            import pandas as pd_

            if isinstance(obj, pd_.DataFrame):
                self._tables[name] = pa.Table.from_pandas(obj)
            else:
                raise TypeError(f"Unsupported object type: {type(obj)}")
        elif schema is not None:
            from ibis.formats.pyarrow import PyArrowSchema

            arrow_schema = PyArrowSchema.from_ibis(schema)
            self._tables[name] = pa.table(
                {field.name: pa.array([], type=field.type) for field in arrow_schema}
            )
        else:
            raise ValueError("One of `obj` or `schema` must be provided")

        self._catalog.register_constant_table(name, self._tables[name])
        return self.table(name)

    def drop_table(
        self,
        name: str,
        /,
        *,
        database: tuple[str, str] | str | None = None,
        force: bool = False,
    ) -> None:
        try:
            del self._tables[name]
        except KeyError:
            if not force:
                raise

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        self._tables[op.name] = op.data.to_pyarrow(op.schema)
        self._catalog.register_constant_table(op.name, self._tables[op.name])

    def _make_memtable_finalizer(self, name: str) -> None:
        return None

    def _get_schema_using_query(self, query: str) -> sch.Schema:
        from ibis.formats.pyarrow import PyArrowSchema

        limited = f"SELECT * FROM ({query}) AS _t LIMIT 0"
        result = self._execute_chalkdf_sql(limited)
        return PyArrowSchema.to_ibis(result.schema)

    @contextlib.contextmanager
    def _safe_raw_sql(self, query, **kwargs):
        result = self._execute_chalkdf_sql(str(query))
        yield result

    def execute(
        self,
        expr: ir.Expr,
        /,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | pd.Series | Any:
        self._run_pre_execute_hooks(expr)
        table = expr.as_table()
        sql = self.compile(table, params=params, limit=limit, **kwargs)

        result_arrow = self._execute_chalkdf_sql(sql)

        from ibis.formats.pandas import PandasData

        result = result_arrow.to_pandas()
        result = PandasData.convert_table(result, table.schema())
        return expr.__pandas_result__(result)

    def _execute_chalkdf_sql(self, sql: str) -> pa.Table:
        """Execute SQL via chalkdf's libchalk engine and return a PyArrow table."""
        now = pa.scalar(datetime.now(timezone.utc), pa.timestamp("us", "UTC"))

        plan = self._chalksql.sql_to_table(self._catalog, self._env, str(sql))

        opts = self._chalktable.CompilationOptions()
        compiled = self._chalktable.CompiledPlan("velox", opts, [plan])

        ctx = self._chalktable.PlanRunContext(
            correlation_id=None,
            environment_id="test",
            deployment_id="test_deployment",
            requester_id="requester_id",
            operation_id="dummy_op",
            execution_timestamp=now,
            is_online=True,
            max_samples=None,
            observed_at_lower_bound=None,
            observed_at_upper_bound=None,
            customer_metadata={},
            shard_id=0,
            extra_attributes={},
            query_context={},
            error_collector=self._chalk_utils.InMemoryErrorCollector(1000),
            metrics_event_collector=self._chalk_metrics.InMemoryMetricsEventCollector(
                1000
            ),
            chalk_metrics=None,
            batch_reporter=None,
            timeline_trace_writer=None,
            plan_metrics_storage_service=None,
            python_context=None,
        )

        query_output = compiled.run(ctx, {}, {"__execution_ts__": now})
        result = query_output.result()
        return pa.Table.from_batches(result.batches)

    def disconnect(self) -> None:
        self._tables.clear()
