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

/*1. Write SQL statements to return data of the following questions:*/
/*For each location, what is the quantity of unique event types?*/
SELECT location, count(distinct(event_type)) num_unique_event_type
FROM dsstudent.join_table jt
GROUP BY location;


/* What are the top 3 locations with the most volumes?*/
SELECT location, sum(volume) total_volume
FROM dsstudent.join_table jt
GROUP BY location
ORDER BY total_volume DESC 
LIMIT 3;

/*2. Write SQL statements to return data of the following questions:*/
/*For each fault severity, what is the quantity of unique locations?*/
SELECT fault_severity, count(distinct(location)) number_of_unique_location
FROM dsstudent.join_table jt
GROUP BY fault_severity;

/* From the query result above, what is the quantity of unique locations with the fault_severity greater than 1?*/
SELECT fault_severity fs, count(distinct(location)) number_of_unique_location
FROM dsstudent.join_table jt
GROUP BY fs
Having fs > 1;

/* 3. Write a SQL query to return the minimum, maximum, average of the field “Age” for each “Attrition” groups from the “hr” database.*/
use hr; 

SELECT attrition,min(age) min_age, max(age) max_age, avg(Age) avg_age
FROM employee e
GROUP BY attrition;

/*4.  Write a SQL query to return the “Attrition”, “Department” and the number of records from the ”hr” database for each group in the “Attrition” and “Department.” 
 * Sort the returned table by the “Attrition” and “Department” fields in ascending order.*/
SELECT attrition, Department, count(*) num_quantity
FROM employee e
GROUP BY attrition, Department
Having attrition IS NOT NULL
ORDER BY attrition ASC, department ASC;


/*5. From Question #4, can you return the results where the “num_quantity” is greater than 100 records?*/
SELECT attrition, Department, count(*) num_quantity
FROM employee e
GROUP BY attrition, Department
Having attrition IS NOT NULL AND num_quantity > 100
ORDER BY attrition ASC, department ASC;
