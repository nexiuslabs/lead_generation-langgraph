-- Dashboards SQL stubs for Feature 8 (Observability)

-- 1) Tenant Overview by Run
SELECT rss.tenant_id, rss.run_id,
       SUM(rss.count_total)        AS total_items,
       SUM(rss.count_success)      AS total_ok,
       SUM(rss.count_error)        AS total_errors,
       MAX(rss.p95_ms)             AS max_p95_ms,
       MAX(rss.p99_ms)             AS max_p99_ms,
       er.status                   AS run_status,
       er.started_at, er.ended_at,
       EXTRACT(EPOCH FROM (er.ended_at - er.started_at))::INT AS run_duration_s
FROM run_stage_stats rss
LEFT JOIN enrichment_runs er USING(run_id, tenant_id)
GROUP BY 1,2, er.status, er.started_at, er.ended_at
ORDER BY rss.run_id DESC;

-- 2) Vendor usage & cost per run
SELECT tenant_id, run_id, vendor, calls, errors, tokens_input, tokens_output, cost_usd
FROM run_vendor_usage
ORDER BY run_id DESC, vendor;

-- 3) Latency per stage (p50/p95/p99)
SELECT tenant_id, run_id, stage, p50_ms, p95_ms, p99_ms
FROM run_stage_stats
ORDER BY run_id DESC, stage;

-- 4) QA samples per run
SELECT run_id, tenant_id, bucket, COUNT(*) AS samples,
       SUM(CASE WHEN result='pass' THEN 1 ELSE 0 END) AS pass_cnt,
       SUM(CASE WHEN result='fail' THEN 1 ELSE 0 END) AS fail_cnt
FROM qa_samples
GROUP BY 1,2,3
ORDER BY run_id DESC, bucket;

-- 5) Apify LinkedIn usage per day
SELECT date_trunc('day', er.started_at) AS day,
       er.tenant_id,
       SUM(rv.calls)               AS apify_calls,
       SUM(rv.errors)              AS apify_errors
FROM run_vendor_usage rv
JOIN enrichment_runs er USING(run_id, tenant_id)
WHERE rv.vendor = 'apify_linkedin'
GROUP BY 1,2
ORDER BY day DESC, tenant_id;
