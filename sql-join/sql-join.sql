use telecom;

Create TEMPORARY TABLE dsstudent.join_table
SELECT 
	t.id, t.location, t.fault_severity,et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
FROM train t
	LEFT OUTER JOIN event_type et ON t.id = et.id
	LEFT OUTER JOIN log_feature lf ON t.id = lf.id
	LEFT OUTER JOIN resource_type rt ON t.id = rt.id
	LEFT OUTER JOIN severity_type st ON t.id = st.id;
	

SELECT * FROM dsstudent.join_table;