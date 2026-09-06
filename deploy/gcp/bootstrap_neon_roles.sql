\set ON_ERROR_STOP on

-- Run this with Neon's direct (non-pooler) owner connection while connected
-- to a database named cacheeconomics. Passwords are requested by psql and are
-- never stored in this repository or in shell history.
SELECT current_database() = 'cacheeconomics' AS correct_database \gset
\if :correct_database
\else
  \echo 'Refusing role bootstrap: connect to the cacheeconomics database first.'
  \quit 2
\endif

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cacheeconomics_migrator') THEN
        CREATE ROLE cacheeconomics_migrator LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cacheeconomics_app') THEN
        CREATE ROLE cacheeconomics_app LOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cacheeconomics_worker') THEN
        CREATE ROLE cacheeconomics_worker LOGIN;
    END IF;
END
$$;

\echo 'Enter a unique generated password for the migration role.'
\password cacheeconomics_migrator
\echo 'Enter a different generated password for the API role.'
\password cacheeconomics_app
\echo 'Enter a third generated password for the worker role.'
\password cacheeconomics_worker

ALTER ROLE cacheeconomics_migrator NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
ALTER ROLE cacheeconomics_app NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
ALTER ROLE cacheeconomics_worker NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOBYPASSRLS;
ALTER ROLE cacheeconomics_app SET statement_timeout = '15s';
ALTER ROLE cacheeconomics_app SET lock_timeout = '5s';
ALTER ROLE cacheeconomics_app SET idle_in_transaction_session_timeout = '30s';
ALTER ROLE cacheeconomics_worker SET statement_timeout = '60s';
ALTER ROLE cacheeconomics_worker SET lock_timeout = '5s';
ALTER ROLE cacheeconomics_worker SET idle_in_transaction_session_timeout = '30s';

-- A Neon-created role can otherwise inherit broad managed-service access.
-- REVOKE reports a harmless warning when no such membership exists.
REVOKE neon_superuser FROM cacheeconomics_migrator, cacheeconomics_app, cacheeconomics_worker;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT CONNECT ON DATABASE cacheeconomics TO cacheeconomics_migrator, cacheeconomics_app, cacheeconomics_worker;
GRANT USAGE, CREATE ON SCHEMA public TO cacheeconomics_migrator;
GRANT USAGE ON SCHEMA public TO cacheeconomics_app, cacheeconomics_worker;

SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolinherit, rolbypassrls
FROM pg_roles
WHERE rolname IN (
    'cacheeconomics_migrator',
    'cacheeconomics_app',
    'cacheeconomics_worker'
)
ORDER BY rolname;
