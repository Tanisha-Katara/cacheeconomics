#!/bin/sh
# Prove that a PostgreSQL backup can be restored into a disposable database.
# The restore target is erased, so the script refuses broad or ambiguous names.
set -eu

: "${SOURCE_DATABASE_URL:?set SOURCE_DATABASE_URL to the database being backed up}"
: "${RESTORE_DATABASE_URL:?set RESTORE_DATABASE_URL to a disposable restore database}"
: "${CACHEECONOMICS_ALLOW_DESTRUCTIVE_RESTORE_TEST:?set to yes after reviewing the restore target}"

if [ "$CACHEECONOMICS_ALLOW_DESTRUCTIVE_RESTORE_TEST" != "yes" ]; then
  echo "refusing restore: CACHEECONOMICS_ALLOW_DESTRUCTIVE_RESTORE_TEST must equal yes" >&2
  exit 2
fi

: "${RESTORE_APP_DATABASE_URL:?set RESTORE_APP_DATABASE_URL for the restricted app role on the restore database}"
: "${RESTORE_WORKER_DATABASE_URL:?set RESTORE_WORKER_DATABASE_URL for the restricted worker role on the restore database}"

source_name=$(psql "$SOURCE_DATABASE_URL" --no-psqlrc --tuples-only --no-align \
  --command "SELECT current_database()")
restore_name=$(psql "$RESTORE_DATABASE_URL" --no-psqlrc --tuples-only --no-align \
  --command "SELECT current_database()")
app_restore_name=$(psql "$RESTORE_APP_DATABASE_URL" --no-psqlrc --tuples-only --no-align \
  --command "SELECT current_database()")
worker_restore_name=$(psql "$RESTORE_WORKER_DATABASE_URL" --no-psqlrc --tuples-only --no-align \
  --command "SELECT current_database()")

case "$restore_name" in
  *_restore_test) ;;
  *)
    echo "refusing restore: target database name must end in _restore_test" >&2
    exit 2
    ;;
esac
if [ "$source_name" = "$restore_name" ]; then
  echo "refusing restore: source and restore databases are the same" >&2
  exit 2
fi
if [ "$app_restore_name" != "$restore_name" ] || [ "$worker_restore_name" != "$restore_name" ]; then
  echo "refusing restore: restricted-role URLs must point to the guarded restore database" >&2
  exit 2
fi

backup_dir=$(mktemp -d "${TMPDIR:-/tmp}/cacheeconomics-restore.XXXXXX")
backup_file="$backup_dir/database.dump"
cleanup() {
  rm -f "$backup_file"
  rmdir "$backup_dir"
}
trap cleanup EXIT HUP INT TERM

fingerprint_sql="SELECT json_build_object(
  'alembic_version', COALESCE((SELECT version_num FROM alembic_version LIMIT 1), ''),
  'organizations', (SELECT count(*) FROM organizations),
  'sources', (SELECT count(*) FROM sources),
  'ingest_events', (SELECT count(*) FROM ingest_events),
  'analyses', (SELECT count(*) FROM analyses)
)::text"

# A row-count match is not enough: a restore that drops ACL entries leaves the
# data present but makes the API and worker unusable. It can also reopen the two
# SECURITY DEFINER functions through PUBLIC. Check the exact minimum grants,
# denials, owners, role flags, and schema access that the migrations establish.
security_sql="SELECT (
  has_schema_privilege('cacheeconomics_app', 'public', 'USAGE')
  AND has_schema_privilege('cacheeconomics_worker', 'public', 'USAGE')
  AND has_table_privilege('cacheeconomics_app', 'public.sources', 'SELECT')
  AND has_table_privilege('cacheeconomics_app', 'public.jobs', 'UPDATE')
  AND has_table_privilege('cacheeconomics_worker', 'public.sources', 'SELECT')
  AND has_table_privilege('cacheeconomics_worker', 'public.analyses', 'INSERT')
  AND has_table_privilege('cacheeconomics_worker', 'public.jobs', 'UPDATE')
  AND NOT has_table_privilege(
    'cacheeconomics_worker', 'public.source_credentials', 'SELECT'
  )
  AND has_function_privilege(
    'cacheeconomics_app',
    'public.authenticate_source_credential(text,text)',
    'EXECUTE'
  )
  AND NOT has_function_privilege(
    'cacheeconomics_worker',
    'public.authenticate_source_credential(text,text)',
    'EXECUTE'
  )
  AND has_function_privilege(
    'cacheeconomics_worker', 'public.lease_analysis_job(text,integer)', 'EXECUTE'
  )
  AND NOT has_function_privilege(
    'cacheeconomics_app', 'public.lease_analysis_job(text,integer)', 'EXECUTE'
  )
  AND NOT EXISTS (
    SELECT 1 FROM pg_tables
     WHERE schemaname = 'public'
       AND tableowner <> 'cacheeconomics_migrator'
  )
  AND NOT EXISTS (
    SELECT 1 FROM pg_proc
     WHERE oid IN (
       'public.authenticate_source_credential(text,text)'::regprocedure,
       'public.lease_analysis_job(text,integer)'::regprocedure
     )
       AND (
         NOT prosecdef
         OR pg_get_userbyid(proowner) <> 'cacheeconomics_migrator'
       )
  )
  AND NOT EXISTS (
    SELECT 1 FROM pg_roles
     WHERE rolname IN ('cacheeconomics_app', 'cacheeconomics_worker')
       AND (rolsuper OR rolbypassrls)
  )
)::text"

source_fingerprint=$(psql "$SOURCE_DATABASE_URL" --no-psqlrc --tuples-only \
  --no-align --set ON_ERROR_STOP=1 --command "$fingerprint_sql")
source_security=$(psql "$SOURCE_DATABASE_URL" --no-psqlrc --tuples-only \
  --no-align --set ON_ERROR_STOP=1 --command "$security_sql")
if [ "$source_security" != "true" ]; then
  echo "source database is missing the required owner, role, or privilege state" >&2
  exit 1
fi

# Owners and ACLs are part of the security model. The restore administrator
# must be able to restore ownership to the pre-provisioned migration role.
pg_dump "$SOURCE_DATABASE_URL" --format=custom --file "$backup_file"

# This is the only destructive operation. It is scoped to a database whose
# server-reported name passed the _restore_test guard above.
psql "$RESTORE_DATABASE_URL" --no-psqlrc --set ON_ERROR_STOP=1 \
  --command "DROP SCHEMA public CASCADE" \
  --command "CREATE SCHEMA public"
pg_restore --dbname "$RESTORE_DATABASE_URL" --exit-on-error "$backup_file"

restore_fingerprint=$(psql "$RESTORE_DATABASE_URL" --no-psqlrc --tuples-only \
  --no-align --set ON_ERROR_STOP=1 --command "$fingerprint_sql")
if [ "$source_fingerprint" != "$restore_fingerprint" ]; then
  echo "restore fingerprint differs from source" >&2
  exit 1
fi

restore_security=$(psql "$RESTORE_DATABASE_URL" --no-psqlrc --tuples-only \
  --no-align --set ON_ERROR_STOP=1 --command "$security_sql")
if [ "$restore_security" != "true" ]; then
  echo "restore is missing the required owner, role, or privilege state" >&2
  exit 1
fi

# Exercise real logins as both restricted runtime roles. RLS may return zero
# rows without tenant context; success here proves the restored ACLs are usable.
psql "$RESTORE_APP_DATABASE_URL" --no-psqlrc --set ON_ERROR_STOP=1 \
  --command "SELECT count(*) FROM public.sources" >/dev/null
psql "$RESTORE_WORKER_DATABASE_URL" --no-psqlrc --set ON_ERROR_STOP=1 \
  --command "SELECT count(*) FROM public.sources" >/dev/null

echo "backup restore, ownership, and restricted-role access verified in disposable database: $restore_name"
