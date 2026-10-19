import asyncio
import platform
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from psycopg.rows import dict_row

from .utils import LINUX, USE_SUDO


@pytest.mark.parametrize(
    ("global_query_wait_timeout", "user_query_wait_timeout", "db_query_wait_timeout"),
    [
        (2, None, None),
        (None, 2, None),
        (200, 2, None),
        (None, None, 2),
        (200, None, 2),
    ],
)
async def test_query_wait_timeout(
    bouncer,
    global_query_wait_timeout: int,
    user_query_wait_timeout: int,
    db_query_wait_timeout: int,
):
    """
    Test of query_wait_timeout. Assumes that the effective timeout supplied is 2.

    Specifically tests that pgbouncer will correctly:
    1. kill a query that has been waiting for longer than effective query_wait_timeout
    2. not kill a query that is waiting, but for less than effective query_wait_timeout
    """

    db_query_wait_timeout_clause = ""
    if db_query_wait_timeout:
        db_query_wait_timeout_clause += f"query_wait_timeout={db_query_wait_timeout}"

    pgbouncer_ini = f"""
    [databases]
    postgres = host={bouncer.pg.host} port={bouncer.pg.port} pool_size=1 {db_query_wait_timeout_clause}

    [pgbouncer]
    listen_addr = {bouncer.host}
    admin_users = pgbouncer
    auth_type = trust
    auth_file = {bouncer.auth_path}
    listen_port = {bouncer.port}
    logfile = {bouncer.log_path}
    pool_mode = transaction
    """
    if global_query_wait_timeout is not None:
        pgbouncer_ini += f"\nquery_wait_timeout={global_query_wait_timeout}"

    if user_query_wait_timeout is not None:
        pgbouncer_ini += f"\n[users]"
        pgbouncer_ini += f"\npuser1 = query_wait_timeout={user_query_wait_timeout}"

    bouncer.default_db = "postgres"
    bouncer.default_user = "puser1"

    with bouncer.run_with_config(pgbouncer_ini):
        conn_1_fut = bouncer.asleep(3)
        await asyncio.sleep(0.1)

        with pytest.raises(psycopg.OperationalError, match=r"query_wait_timeout"):
            bouncer.test()
        await conn_1_fut

        conn_1_fut = bouncer.asleep(1)
        await asyncio.sleep(0.1)
        bouncer.test()
        await conn_1_fut


def test_server_lifetime(pg, bouncer):
    bouncer.admin(f"set server_lifetime=2")

    bouncer.test()
    assert pg.connection_count() == 1
    time.sleep(3)
    assert pg.connection_count() == 0
    bouncer.test()


def test_server_lifetime_per_pool(pg, bouncer):
    bouncer.test(dbname="p9")
    assert pg.connection_count() == 1
    time.sleep(3)
    assert pg.connection_count() == 0
    bouncer.test(dbname="p9")


def test_server_idle_timeout(pg, bouncer):
    bouncer.admin(f"set server_idle_timeout=2")

    bouncer.test()
    assert pg.connection_count() == 1
    time.sleep(3)
    assert pg.connection_count() == 0
    bouncer.test()


def test_user_idle_transaction_timeout_negative(bouncer):
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session

        [users]
        puser1 = pool_mode=transaction idle_transaction_timeout=6
    """

    # while configured to be in statement pooling mode
    with (
        bouncer.run_with_config(config),
        bouncer.transaction(dbname="postgres", user="puser1") as cur,
    ):
        time.sleep(3)
        cur.execute("select 1")


def test_user_idle_transaction_timeout_override_global(bouncer):
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session
        idle_transaction_timeout=100000

        [users]
        puser1 = pool_mode=transaction idle_transaction_timeout=1
    """

    # while configured to be in statement pooling mode
    with (
        bouncer.run_with_config(config),
        bouncer.transaction(dbname="postgres", user="puser1") as cur,
        bouncer.log_contains(r"idle transaction timeout"),
    ):
        time.sleep(3)
        with pytest.raises(
            psycopg.OperationalError,
            match=r"idle transaction timeout|Software caused connection abort|server closed the connection unexpectedly",
        ):
            cur.execute("select 1")


def test_user_idle_transaction_timeout(bouncer):
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session

        [users]
        puser1 = pool_mode=transaction idle_transaction_timeout=1
    """

    # while configured to be in statement pooling mode
    with (
        bouncer.run_with_config(config),
        bouncer.transaction(dbname="postgres", user="puser1") as cur,
        bouncer.log_contains(r"idle transaction timeout"),
    ):
        time.sleep(3)
        with pytest.raises(
            psycopg.OperationalError,
            match=r"idle transaction timeout|Software caused connection abort|server closed the connection unexpectedly",
        ):
            cur.execute("select 1")


def test_user_query_timeout_override_global(bouncer):
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session
        query_timeout=100000

        [users]
        puser1 = pool_mode=statement query_timeout=1
    """

    # while configured to be in statement pooling mode
    with (
        bouncer.run_with_config(config),
        bouncer.log_contains(r"query timeout"),
        pytest.raises(
            psycopg.OperationalError,
            match=r"query timeout|server closed the connection unexpectedly",
        ),
    ):
        bouncer.sleep(5, user="puser1", dbname="postgres")


def test_user_query_timeout_negative(bouncer):
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session

        [users]
        puser1 = pool_mode=statement query_timeout=10
    """

    # while configured to be in statement pooling mode
    with bouncer.run_with_config(config):
        bouncer.sleep(5, user="puser1", dbname="postgres")


def test_user_query_timeout(bouncer):
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session

        [users]
        puser1 = pool_mode=statement query_timeout=1
    """

    # while configured to be in statement pooling mode
    with (
        bouncer.run_with_config(config),
        bouncer.log_contains(r"query timeout"),
        pytest.raises(
            psycopg.OperationalError,
            match=r"query timeout|server closed the connection unexpectedly",
        ),
    ):
        bouncer.sleep(5, user="puser1", dbname="postgres")


def test_query_timeout(bouncer):
    bouncer.admin(f"set query_timeout=1")

    with (
        bouncer.log_contains(r"query timeout"),
        pytest.raises(
            psycopg.OperationalError,
            match=r"query timeout|server closed the connection unexpectedly",
        ),
    ):
        bouncer.sleep(5)


def test_user_level_idle_client_timeout_negative(bouncer):
    """Test user level client_idle_timeout correctly closes connection."""
    bouncer.admin("set pool_mode=transaction")
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        auth_type = trust
        admin_users = pgbouncer
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        auth_dbname = postgres
        pool_mode = session

        [users]
        puser1 = pool_mode=transaction client_idle_timeout=2
    """

    with bouncer.run_with_config(config):
        bouncer.admin("RELOAD")
        with bouncer.cur(dbname="postgres", user="puser1") as cur:
            cur.execute("SELECT 1")
            with bouncer.log_contains(r"client_idle_timeout"):
                time.sleep(3)
                with pytest.raises(
                    psycopg.OperationalError,
                    match=r"client_idle_timeout|Software caused connection abort|server closed the connection unexpectedly",
                ):
                    cur.execute("SELECT 1")


def test_user_level_idle_client_timeout(bouncer):
    """Test that user level client_idle_timeout allows connection to stay open."""
    bouncer.admin("set pool_mode=transaction")
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        auth_type = trust
        admin_users = pgbouncer
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        auth_dbname = postgres
        pool_mode = session

        [users]
        puser1 = pool_mode=transaction client_idle_timeout=6
    """

    with bouncer.run_with_config(config):
        bouncer.admin("RELOAD")
        with bouncer.cur(dbname="postgres", user="puser1") as cur:
            cur.execute("SELECT 1")
            time.sleep(3)
            cur.execute("SELECT 1")


def test_user_level_idle_client_timeout_override(bouncer):
    """Test that user level client_idle_timeout overrides global level setting."""
    bouncer.admin("set pool_mode=transaction")
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        auth_type = trust
        admin_users = pgbouncer
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        auth_dbname = postgres
        pool_mode = session
        client_idle_timeout=1000000

        [users]
        puser1 = pool_mode=transaction client_idle_timeout=2
    """

    with bouncer.run_with_config(config):
        bouncer.admin("RELOAD")
        with bouncer.cur(dbname="postgres", user="puser1") as cur:
            cur.execute("SELECT 1")
            with bouncer.log_contains(r"client_idle_timeout"):
                time.sleep(3)
                with pytest.raises(
                    psycopg.OperationalError,
                    match=r"client_idle_timeout|Software caused connection abort|server closed the connection unexpectedly",
                ):
                    cur.execute("SELECT 1")


def test_transaction_timeout_user_overide_global(bouncer):
    """
    Test user level transaction timeout correctly overrides global setting.

    Procedure:
        - Start pgbouncer with config that has
          user level transaction timeout of 2 seconds for user psuser1
          and a global level setting of 10 seconds
        - Start transaction with user puser1
        - Wait 1 second
        - Test that empty query works
        - Wait 2 seconds
        - Test that empty query raises psycopg.OperationalError
    """
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        transaction_timeout=10
        pool_mode = session

        [users]
        puser1 = pool_mode=transaction transaction_timeout=2
    """

    with (
        bouncer.run_with_config(config),
        bouncer.transaction(dbname="postgres", user="puser1") as cur,
    ):
        time.sleep(1)
        cur.execute("")
        with (
            bouncer.log_contains(r"transaction timeout"),
            pytest.raises(
                psycopg.OperationalError,
                match=r"transaction timeout|Software caused connection abort",
            ),
        ):
            time.sleep(2)
            cur.execute("")


def test_transaction_timeout_user(bouncer):
    """
    Test user level transaction timeout.

    Procedure:
        - Start pgbouncer with config that has
          user level transaction timeout of 2 seconds for user psuser1.
        - Start transaction with user puser1
        - Wait 1 second
        - Test that empty query works
        - Wait 2 seconds
        - Test that empty query raises psycopg.OperationalError
    """
    config = f"""
        [databases]
        postgres = host={bouncer.pg.host} port={bouncer.pg.port}

        [pgbouncer]
        listen_addr = {bouncer.host}
        admin_users = pgbouncer
        auth_type = trust
        auth_file = {bouncer.auth_path}
        listen_port = {bouncer.port}
        logfile = {bouncer.log_path}
        pool_mode = session

        [users]
        puser1 = pool_mode=transaction transaction_timeout=2
    """

    with (
        bouncer.run_with_config(config),
        bouncer.transaction(dbname="postgres", user="puser1") as cur,
    ):
        time.sleep(1)
        cur.execute("")
        with (
            bouncer.log_contains(r"transaction timeout"),
            pytest.raises(
                psycopg.OperationalError,
                match=r"transaction timeout|Software caused connection abort",
            ),
        ):
            time.sleep(2)
            cur.execute("")


def test_transaction_timeout(bouncer):
    """
    Test pgbouncer level transaction timeout.

    Procedure:
        - Set pool_mode=transaction in admin console (default is statement)
        - Set transaction_timeout=2
        - start transaction.
        - Wait one second
        - Execute empty query. Test that no error is raised
        - Wait 2 seconds
        - Execute emtpty query. Test that psycopg.OperationalError is raised
    """
    bouncer.admin("SET pool_mode=transaction")
    bouncer.admin("SET transaction_timeout=2")

    with bouncer.transaction() as cur:
        time.sleep(1)
        cur.execute("")
        with bouncer.log_contains(r"transaction timeout"):
            time.sleep(2)
            with pytest.raises(
                psycopg.OperationalError,
                match=r"transaction timeout|Software caused connection abort",
            ):
                cur.execute("")


def test_idle_transaction_timeout(bouncer):
    bouncer.admin(f"set pool_mode=transaction")
    bouncer.admin(f"set idle_transaction_timeout=2")

    with (
        bouncer.transaction() as cur,
        bouncer.log_contains(r"idle transaction timeout"),
    ):
        time.sleep(3)
        with pytest.raises(
            psycopg.OperationalError,
            match=r"idle transaction timeout|Software caused connection abort|server closed the connection unexpectedly",
        ):
            cur.execute("select 1")

    # test for GH issue #125
    with bouncer.transaction() as cur:
        cur.execute("select pg_sleep(2)").fetchone()
        time.sleep(1)
        cur.execute("select 1")


def test_client_idle_timeout(bouncer):
    bouncer.admin(f"set client_idle_timeout=2")

    with bouncer.cur() as cur:
        cur.execute("select 1")
        with bouncer.log_contains(r"client_idle_timeout"):
            time.sleep(3)
            with pytest.raises(
                psycopg.OperationalError,
                match=r"client_idle_timeout|Software caused connection abort|server closed the connection unexpectedly",
            ):
                cur.execute("select 1")


def test_pool_idle_timeout(pg, bouncer):
    """Test that the pool closes server connections after being idle."""
    bouncer.admin("set pool_idle_timeout=1")
    bouncer.admin("set server_idle_timeout=1")

    bouncer.test()
    assert pg.connection_count() == 1

    # The non-admin pool exists after connecting. Column 0 of SHOW POOLS is
    # the database, and there's always an admin ("pgbouncer") pool.
    pools_before = bouncer.admin("show pools")
    print("pools before idle timeout:", pools_before)
    assert any(row[0] != "pgbouncer" for row in pools_before)

    with bouncer.log_contains(r"cleaning up idle pool"):
        time.sleep(3)

    # The idle pool is gone entirely, not just its server connections.
    pools_after = bouncer.admin("show pools")
    print("pools after idle timeout:", pools_after)
    assert all(row[0] == "pgbouncer" for row in pools_after)

    assert pg.connection_count() == 0

    bouncer.test()
    assert pg.connection_count() == 1


def test_pool_idle_timeout_ignores_min_pool_size(pg, bouncer):
    """A pool configured with min_pool_size must not be reaped by
    pool_idle_timeout, even when it currently has zero live server
    connections.

    We use a forced user together with min_pool_size, so the janitor
    proactively creates the pool and keeps trying to maintain backend
    connections even without any clients. The backend database does not
    exist, so every connection attempt fails, which leaves the pool with
    zero connected servers. On top of that, once a connect fails there is a
    server_login_retry backoff during which no new connection is launched,
    so the pool's last_active_time stops getting bumped. Without an
    exemption for min_pool_size, cleanup_inactive_pools then reaps the pool.
    """
    bouncer.write_ini("pool_idle_timeout = 1")
    bouncer.write_ini(
        "[databases]\n"
        "min_pool_size_dead = port=6666 host=127.0.0.1"
        " dbname=non_existing_pg_db min_pool_size=3 user=pswcheck"
    )
    bouncer.admin("reload")

    with bouncer.log_contains(r"cleaning up idle pool.*min_pool_size_dead", times=0):
        time.sleep(3)


async def test_server_login_retry(pg, bouncer):
    bouncer.admin(f"set query_timeout=10")
    bouncer.admin(f"set server_login_retry=3")

    # Disable tls to get more consistent timings
    bouncer.admin("set server_tls_sslmode = disable")

    pg.stop()
    if platform.system() in ("FreeBSD", "Windows"):
        # XXX: For some reason FreeBSD and Windows logs don't contain connect
        # failed. For now we simply remove this check. But this warrants
        # further investigation.
        await asyncio.gather(
            bouncer.atest(connect_timeout=10),
            pg.delayed_start(1),
        )
    else:
        with bouncer.log_contains("connect failed"):
            await asyncio.gather(
                bouncer.atest(connect_timeout=10),
                pg.delayed_start(1),
            )


def test_server_connect_timeout_establish(pg, bouncer):
    pg.configure("pre_auth_delay to '5s'")
    pg.reload()
    bouncer.admin("set query_timeout=3")
    bouncer.admin("set server_connect_timeout=2")
    with (
        bouncer.log_contains(r"closing because: connect timeout"),
        pytest.raises(psycopg.errors.OperationalError, match="query_timeout"),
    ):
        bouncer.test(connect_timeout=10)


@pytest.mark.skipif("not USE_SUDO")
def test_server_connect_timeout_drop_traffic(pg, bouncer):
    bouncer.admin("set query_timeout=3")
    bouncer.admin("set server_connect_timeout=2")
    with (
        bouncer.log_contains(r"closing because: connect failed"),
        pg.drop_traffic(),
        pytest.raises(psycopg.errors.OperationalError, match="query_timeout"),
    ):
        bouncer.test(connect_timeout=10)


@pytest.mark.skipif("not USE_SUDO")
@pytest.mark.skipif(
    "platform.system() != 'Linux'", reason="tcp_user_timeout is only supported on Linux"
)
def test_tcp_user_timeout(pg, bouncer):
    bouncer.admin("set tcp_user_timeout=1000")
    bouncer.admin("set query_timeout=5")
    # Make PgBouncer cache a connection to Postgres
    bouncer.test()
    # without tcp_user_timeout, you get a different error message
    # about "query timeout" instead
    with (
        bouncer.log_contains(r"closing because: server conn crashed?"),
        pg.reject_traffic(),
        pytest.raises(
            psycopg.OperationalError,
            match=r"query timeout|Software caused connection abort|server closed the connection unexpectedly",
        ),
    ):
        bouncer.test(connect_timeout=10)


@pytest.mark.skipif("not USE_SUDO")
async def test_server_check_delay(pg, bouncer):
    bouncer.admin("set server_check_delay=2")
    bouncer.admin("set server_login_retry=3")
    bouncer.admin("set query_timeout=10")
    with pg.drop_traffic():
        await asyncio.sleep(3)
        query_task = bouncer.atest(connect_timeout=10)

        # We wait for 1 second to show that the query is blocked while traffic
        # is dropped.
        done, pending = await asyncio.wait([query_task], timeout=1)
        assert done == set()
        assert pending == {query_task}
    await query_task


@pytest.mark.skipif("not USE_SUDO")
def test_cancel_wait_timeout(pg, bouncer):
    bouncer.admin("set cancel_wait_timeout=1")
    with bouncer.cur() as cur, ThreadPoolExecutor(max_workers=2) as pool:
        query = pool.submit(cur.execute, "select pg_sleep(3)")

        time.sleep(1)

        with (
            pg.drop_traffic(),
            bouncer.log_contains(r"closing because: cancel_wait_timeout"),
        ):
            cancel = pool.submit(cur.connection.cancel)
            cancel.result()

        query.result()


def test_query_timeout_when_no_active_query(bouncer):
    """
    When there's no active query (query_start == 0), query_timeout should not apply.
    But the current implementation incorrectly uses request_time as fallback,
    causing idle connections to be disconnected.
    """
    bouncer.admin("set query_timeout=2")
    bouncer.admin("set pool_mode=transaction")
    with bouncer.cur() as cur:
        cur.execute("BEGIN")
        cur.execute("SELECT 1")

        # Wait longer than query_timeout while idle
        time.sleep(3)

        # This should work since no query is active, but will fail due to the bug
        cur.execute("SELECT 2")
        result = cur.fetchone()
        assert result[0] == 2


# tcp_socket_buffer pins the client sockets' receive buffer, so the FIN of a
# client that hung up always fits behind the unread rest of its query, whatever
# the kernel's default buffer size.
WAITING_HANGUP_CONFIG = """
[databases]
postgres = host={pg_host} port={pg_port} pool_size=1

[pgbouncer]
listen_addr = {host}
admin_users = pgbouncer
auth_type = trust
auth_file = {auth_path}
listen_port = {port}
logfile = {log_path}
pool_mode = transaction
query_wait_timeout = 60
tcp_socket_buffer = 65536
"""


def _waiting_hangup_config(bouncer):
    return WAITING_HANGUP_CONFIG.format(
        pg_host=bouncer.pg.host,
        pg_port=bouncer.pg.port,
        host=bouncer.host,
        auth_path=bouncer.auth_path,
        port=bouncer.port,
        log_path=bouncer.log_path,
    )


def _queue_query_then_keep_socket(bouncer, query="select 1"):
    """Log in over a raw socket and send a query without reading its result.

    The caller closes the socket to act like a client whose own deadline
    expired while PgBouncer still had it queued. A query larger than PgBouncer's
    packet buffer leaves the rest of itself in the kernel when the client is
    paused, which is how production clients lost servers: on activation
    PgBouncer read and forwarded that rest before it read the EOF.
    """
    sock = socket.create_connection((bouncer.host, bouncer.port), timeout=10)
    params = b"user\0puser1\0database\0postgres\0\0"
    sock.sendall(struct.pack("!ii", 8 + len(params), 196608) + params)
    received = b""
    while True:
        chunk = sock.recv(8192)
        assert chunk, "PgBouncer closed the login"
        received += chunk
        while len(received) >= 5:
            kind = received[:1]
            length = struct.unpack("!i", received[1:5])[0]
            if len(received) < 1 + length:
                break
            assert kind != b"E", received
            received = received[1 + length :]
            if kind == b"Z":
                payload = query.encode() + b"\0"
                sock.sendall(b"Q" + struct.pack("!i", 4 + len(payload)) + payload)
                return sock


def _hang_up(sock):
    """Close a queued client's socket once PgBouncer's kernel holds its FIN.

    A half-close leaves FIN_WAIT1 when PgBouncer's kernel acknowledges the FIN
    (FIN_WAIT2), or when PgBouncer has already dropped the client in reply
    (TIME_WAIT, CLOSING or CLOSE). Either way the FIN sits in PgBouncer's
    receive queue behind any unread rest of the query, which is what a client
    that gave up while queued looks like, with no assumption about how long
    delivery takes.
    """
    tcp_fin_wait1 = 4  # linux/tcp_states.h
    try:
        sock.shutdown(socket.SHUT_WR)
        for _ in range(200):
            if sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 8)[0] != tcp_fin_wait1:
                return
            time.sleep(0.01)
        raise AssertionError("PgBouncer's kernel did not acknowledge the client's FIN")
    finally:
        sock.close()


def _wait_for_waiting_clients(bouncer, count):
    for _ in range(100):
        pools = bouncer.admin("SHOW POOLS", row_factory=dict_row)
        waiting = sum(pool["cl_waiting"] for pool in pools if pool["database"] == "postgres")
        if waiting == count:
            return
        time.sleep(0.05)
    raise AssertionError(f"expected {count} waiting clients, saw {waiting}")


@pytest.mark.skipif(not LINUX, reason="needs TCP_INFO and POLLRDHUP")
def test_released_server_skips_waiting_client_that_hung_up(bouncer):
    """A server released to the queue must not go to a client that gave up.

    Before the fix PgBouncer activated the oldest waiting client, forwarded the
    part of its query still in the kernel to the server, then read its EOF and
    closed that server mid-message. A live client queued behind it must get the
    same server.
    """
    bouncer.default_db = "postgres"
    bouncer.default_user = "puser1"
    with bouncer.run_with_config(_waiting_hangup_config(bouncer)):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with bouncer.transaction() as cur:
                cur.execute("select pg_backend_pid()")
                pid = cur.fetchone()[0]
                # Larger than PgBouncer's 4 kB packet buffer, so part of it is
                # still in the kernel when the client hangs up.
                gone = _queue_query_then_keep_socket(
                    bouncer, "select length('" + "x" * 6144 + "')"
                )
                _wait_for_waiting_clients(bouncer, 1)
                live = pool.submit(bouncer.sql_value, "select pg_backend_pid()")
                _wait_for_waiting_clients(bouncer, 2)
                _hang_up(gone)
            assert live.result(timeout=10) == pid
        assert bouncer.sql_value("select pg_backend_pid()") == pid
        servers = bouncer.admin("SHOW SERVERS", row_factory=dict_row)
        assert [server["database"] for server in servers] == ["postgres"]


@pytest.mark.skipif(not LINUX, reason="needs TCP_INFO and POLLRDHUP")
def test_resumed_pool_skips_waiting_client_that_hung_up(bouncer):
    """The janitor must not launch or hand a server to a client that gave up.

    PAUSE closes the pool's servers, so a query sent while paused waits. On
    RESUME the janitor walks the queue; before the fix it launched a server for
    a client that had already hung up, handed it the abandoned query and then
    closed it on that client's EOF.
    """
    bouncer.default_db = "postgres"
    bouncer.default_user = "puser1"
    with bouncer.run_with_config(_waiting_hangup_config(bouncer)):
        bouncer.test()
        bouncer.admin("PAUSE postgres")
        resumed = False
        try:
            gone = _queue_query_then_keep_socket(bouncer)
            _wait_for_waiting_clients(bouncer, 1)
            _hang_up(gone)
            # The queue empties only after the janitor has dealt with the
            # client: before the fix that means a server was launched for it
            # and logged first, so the counts below hold once it is empty.
            with (
                bouncer.log_contains(r"client hung up while waiting for a server", times=1),
                bouncer.log_contains(r"new connection to server", times=0),
            ):
                bouncer.admin("RESUME postgres")
                resumed = True
                _wait_for_waiting_clients(bouncer, 0)
        finally:
            if not resumed:
                bouncer.admin("RESUME postgres")
        bouncer.test()
