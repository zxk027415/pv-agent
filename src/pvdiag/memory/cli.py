"""A4 memory options: read-only, explicitly revision-frozen views; DSNs never on the command line."""

from contextlib import asynccontextmanager
from pathlib import Path


def add_memory_arguments(parser):
    parser.add_argument("--memory-dsn-env", default="PVDIAG_MEMORY_DSN")
    parser.add_argument("--memory-schema", default="pvdiag_memory")
    parser.add_argument(
        "--memory-revision", type=int, help="A4 only: frozen memory revision to recall from"
    )
    parser.add_argument(
        "--memory-embedding-path", type=Path, help="A4 only: reviewed local BGE snapshot"
    )


@asynccontextmanager
async def memory_from_args(args):
    a4 = getattr(args, "ablation", None) == "A4"
    given = args.memory_revision is not None or args.memory_embedding_path is not None
    if not a4:
        if given:
            raise ValueError("memory options are only valid with --ablation A4")
        yield None
        return
    if args.memory_revision is None or args.memory_embedding_path is None:
        raise ValueError("A4 requires --memory-revision and --memory-embedding-path")
    from pvdiag.agent.sessions import connection_setting
    from pvdiag.memory.embedding import MemoryEmbedding
    from pvdiag.memory.store import PgVectorStore

    dsn = connection_setting(args.memory_dsn_env)
    async with PgVectorStore.open(
        dsn, MemoryEmbedding(args.memory_embedding_path), schema=args.memory_schema
    ) as store:
        yield await store.view(revision=args.memory_revision)
