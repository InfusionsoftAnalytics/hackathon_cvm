--A snapshot of ongoing CVM, active apps through the whole month that are at least 30 days old
SELECT *

FROM `infusionsoft-looker-poc.analytics.mvc_daily_table`
where date>='2020-06-01' and date<='2020-06-30' and (lost_revenue_date is null OR lost_revenue_date > '2020-06-30') and paying_customer_date<'2020-05-01'
order by date asc