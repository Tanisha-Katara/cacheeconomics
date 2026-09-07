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

-- Interactive password entry is the safe default. A trusted automation process
-- can opt in to environment-backed values without placing passwords in command
-- arguments or this file.
\if :{?cacheeconomics_passwords_from_environment}
  \getenv cacheeconomics_migrator_password CACHEECONOMICS_MIGRATOR_PASSWORD
  \getenv cacheeconomics_app_password CACHEECONOMICS_APP_PASSWORD
  \getenv cacheeconomics_worker_password CACHEECONOMICS_WORKER_PASSWORD
  ALTER ROLE cacheeconomics_migrator PASSWORD :'cacheeconomics_migrator_password';
  ALTER ROLE cacheeconomics_app PASSWORD :'cacheeconomics_app_password';
  ALTER ROLE cacheeconomics_worker PASSWORD :'cacheeconomics_worker_password';
\else
  \echo 'Enter a unique generated password for the migration role.'
  \password cacheeconomics_migrator
  \echo 'Enter a different generated password for the API role.'
  \password cacheeconomics_app
  \echo 'Enter a third generated password for the worker role.'
  \password cacheeconomics_worker
\endif

-- Neon's managed owner is deliberately not a PostgreSQL superuser, so it may
-- not spell out NOSUPERUSER or NOBYPASSRLS in ALTER ROLE. Roles created here
-- through SQL already default to both; the fail-closed assertion below proves
-- the attributes instead of relying on the default silently.
ALTER ROLE cacheeconomics_migrator NOCREATEDB NOCREATEROLE NOINHERIT;
ALTER ROLE cacheeconomics_app NOCREATEDB NOCREATEROLE NOINHERIT;
ALTER ROLE cacheeconomics_worker NOCREATEDB NOCREATEROLE NOINHERIT;
ALTER ROLE cacheeconomics_app SET statement_timeout = '15s';
ALTER ROLE cacheeconomics_app SET lock_timeout = '5s';
ALTER ROLE cacheeconomics_app SET idle_in_transaction_session_timeout = '30s';
ALTER ROLE cacheeconomics_worker SET statement_timeout = '60s';
ALTER ROLE cacheeconomics_worker SET lock_timeout = '5s';
ALTER ROLE cacheeconomics_worker SET idle_in_transaction_session_timeout = '30s';

-- Roles created through SQL do not receive Neon's managed neon_superuser
-- membership. The managed owner cannot revoke that protected role, even as a
-- harmless no-op, so prove non-membership below and abort if Neon ever changes
-- the SQL-created-role behavior.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_roles
        WHERE rolname IN (
            'cacheeconomics_migrator',
            'cacheeconomics_app',
            'cacheeconomics_worker'
        )
        AND (rolsuper OR rolcreatedb OR rolcreaterole OR rolinherit OR rolbypassrls)
    ) THEN
        RAISE EXCEPTION 'cacheeconomics runtime database roles are over-privileged';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_auth_members memberships
        JOIN pg_roles granted_role ON granted_role.oid = memberships.roleid
        JOIN pg_roles member_role ON member_role.oid = memberships.member
        WHERE granted_role.rolname = 'neon_superuser'
        AND member_role.rolname IN (
            'cacheeconomics_migrator',
            'cacheeconomics_app',
            'cacheeconomics_worker'
        )
    ) THEN
        RAISE EXCEPTION 'cacheeconomics runtime roles must not inherit neon_superuser';
    END IF;
END
$$;

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
