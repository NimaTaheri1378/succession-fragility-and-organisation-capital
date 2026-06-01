-- Template only. Do not run until schema_map.json identifies the exact tables and columns.
-- Goal: compact point-in-time executive/person-role histories for U.S. listed firms.
-- Filters must be pushed server-side and date predicates must remain index-friendly.

select
  *
from {boardex_role_table}
where {date_column} between :start_date and :end_date;
