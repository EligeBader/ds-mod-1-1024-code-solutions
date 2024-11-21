/* Exercice 2 - SQL-Coomon_Clauses */

Select lf.id, lf.log_feature log, lf.volume vol
FROM log_feature lf;


/* Sorting */

/*Write a SQL query to return the first 5 rows of “id”, “resource_type” and sorted by ”id” & "resource_type" columns in ascending order.*/
SELECT id,resource_type
FROM resource_type rt
ORDER BY id, resource_type ASC
LIMIT 5;

/*Write a SQL query to return the last 5 rows of “id”, “resource_type” and sorted by ”id” column in descending order.*/
SELECT id, resource_type
FROM resource_type rt
ORDER BY id DESC 
LIMIT 5;

/* Write a SQL query to return 5 rows of “id”, “resource_type” and sorted by ”id” column in ascending order first,
 *  then sorted by “resource_type” column in a descending order.*/
SELECT id,resource_type
FROM resource_type rt
ORDER BY id ASC, resource_type DESC
LIMIT 5;

/* Count/Distinct*/

SELECT 
  COUNT(*) AS total_rows,
  COUNT(DISTINCT id) AS unique_ids,
  COUNT(DISTINCT severity_type) AS unique_severity_types
FROM 
  severity_type;
  
 /* Where filtering*/
  
Select id, log_feature, volume 
FROM log_feature lf
WHERE log_feature = 'Feature 201' AND Volume Between 100 AND 300
ORDER BY volume ;
