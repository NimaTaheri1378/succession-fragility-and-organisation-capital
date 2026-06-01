-- Template only. Monthly CRSP extraction should retain only needed columns.
-- Include delisting-aware return logic in Python after pulling monthly return and delist fields.

select
  m.permno,
  m.permco,
  m.date,
  m.ret,
  m.retx,
  m.prc,
  m.shrout,
  n.shrcd,
  n.exchcd,
  n.siccd,
  n.ticker,
  n.comnam,
  n.cusip,
  n.ncusip
from crsp.msf as m
join crsp.msenames as n
  on m.permno = n.permno
 and n.namedt <= m.date
 and m.date <= coalesce(n.nameendt, '9999-12-31')
where m.date between :start_date and :end_date
  and n.shrcd in (10, 11)
  and n.exchcd in (1, 2, 3);
