use telecom;


SELECT id, log_feature, volume, 
		CASE 
			WHEN volume < 100 Then "low"
			WHEN volume > 500 Then "large"
			ELSE "medium"
		END volume_1
FROM log_feature;
		

CREATE TEMPORARY TABLE dsstudent.volume
SELECT id, log_feature, volume, 
		CASE 
			WHEN volume < 100 Then "low"
			WHEN volume > 500 Then "large"
			ELSE "medium"
		END volume_1
FROM log_feature;

SELECT * FROM dsstudent.volume;

SELECT 
  volume_1,
  COUNT(*) AS value_counts
FROM 
  dsstudent.volume
GROUP BY 
  volume_1;
  
 /* Hourly Rate */
SELECT EmployeeNumber, HourlyRate, 
		CASE 
			WHEN HourlyRate < 40 Then "low hourly rate"
			WHEN HourlyRate > 80 THEN "high hourly rate"
			ELSE "medium hourly rate"
		END HourlyRate_1
FROM employee;


Select * from employee;

SELECT Gender, 
		CASE 
			WHEN Gender = "Female" Then 0
			WHEN Gender = "Male" THEN 1
		END Gender_1
FROM employee;
