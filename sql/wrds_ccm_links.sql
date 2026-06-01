-- Template only. Use valid link windows and document linktype/linkprim choices.

select
  gvkey,
  lpermno as permno,
  linkdt,
  linkenddt,
  linktype,
  linkprim
from {ccm_link_table}
where linktype in ('LU', 'LC')
  and linkprim in ('P', 'C')
  and lpermno is not null;
