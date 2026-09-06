\set ON_ERROR_STOP on

-- Local-development credentials only. Production creates these roles and
-- passwords through its secret manager and infrastructure tooling.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cacheeconomics_migrator') THEN
        CREATE ROLE cacheeconomics_migrator LOGIN PASSWORD 'migrator-dev-only';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cacheeconomics_app') THEN
        CREATE ROLE cacheeconomics_app LOGIN PASSWORD 'app-dev-only';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cacheeconomics_worker') THEN
        CREATE ROLE cacheeconomics_worker LOGIN PASSWORD 'worker-dev-only';
    END IF;
END
$$;

ALTER ROLE cacheeconomics_migrator NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
ALTER ROLE cacheeconomics_app NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
ALTER ROLE cacheeconomics_worker NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
ALTER ROLE cacheeconomics_app SET statement_timeout = '15s';
ALTER ROLE cacheeconomics_app SET lock_timeout = '5s';
ALTER ROLE cacheeconomics_app SET idle_in_transaction_session_timeout = '30s';
ALTER ROLE cacheeconomics_worker SET statement_timeout = '60s';
ALTER ROLE cacheeconomics_worker SET lock_timeout = '5s';
ALTER ROLE cacheeconomics_worker SET idle_in_transaction_session_timeout = '30s';

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE cacheeconomics TO cacheeconomics_migrator, cacheeconomics_app, cacheeconomics_worker;
GRANT USAGE, CREATE ON SCHEMA public TO cacheeconomics_migrator;
GRANT USAGE ON SCHEMA public TO cacheeconomics_app;
GRANT USAGE ON SCHEMA public TO cacheeconomics_worker;
