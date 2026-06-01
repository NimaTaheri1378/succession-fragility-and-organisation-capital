-- Template only. Annual Compustat controls with availability lags handled downstream.

select
  gvkey,
  datadate,
  fyear,
  at,
  ceq,
  seq,
  txditc,
  pstk,
  pstkrv,
  pstkl,
  sale,
  cogs,
  xsga,
  xrd,
  capx,
  dltt,
  dlc,
  che,
  ni
from {comp_funda_table}
where indfmt = 'INDL'
  and datafmt = 'STD'
  and popsrc = 'D'
  and consol = 'C'
  and datadate between :start_date and :end_date;
