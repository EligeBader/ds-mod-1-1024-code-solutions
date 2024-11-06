use hr;

/* 1. Write a SQL query to return the “Attrition”, “Department”, “Gender”, “JobRole”, ‘MonthlyIncome” from “hr” database.*/

SELECT Attrition, Department, Gender, JobRole, MonthlyIncome
FROM employee e;

/* 2. Write a SQL query to return the “Attrition”, “Department”, “Gender”, “JobRole”, ‘MonthlyIncome” from “hr” database
* with column alias (you can refer to column alias names in the following picture.)*/

SELECT Attrition attr, Department dep, Gender sex, JobRole title, MonthlyIncome m_income
FROM employee e;

/* 3. Write a SQL query to return the “MaritalStatus”, “OverTime”, “TotalWorkingYears” from “hr” database 
 * with column alias (you can refer to column alias names in the following picture.)*/
SELECT MaritalStatus marital_status, OverTime over_time, TotalWorkingYears total_working_years
FROM employee e;

/*4. Write a SQL query to return the top 3 highest monthly income and their job roles from the “hr” database.*/
SELECT JobRole, MonthlyIncome m_income
FROM employee e
ORDER BY m_income Desc  
LIMIT 3;

/* 5. Write a SQL query to return the top 3 longest total working years and the departments from the “hr” database.*/
SELECT Department, TotalWorkingYears total_working_years
FROM employee e
ORDER BY total_working_years Desc  
LIMIT 3;


/* 6. Write a SQL query to return the top 5 youngest age and the departments from the “hr” database.*/
SELECT Department, age
FROM employee e
ORDER BY age ASC  
LIMIT 5;

/*7. Write a SQL query to return the unique values of the field “Department” from the “hr” database.*/
SELECT DISTINCT (Department)
FROM employee e;

/* 8. Write a SQL query to return the total rows of this table and use a column alias to represent the total rows from the “hr” database.*/
SELECT COUNT(*) tota_rows
FROM employee e;


/*9. Write a SQL query to return the quantity of unique values in the field “JobRole” 
 * and use a column alias to represent the quantity of unique values from the “hr” database.*/
SELECT COUNT(DISTINCT(JobRole)) number_unique_jobrole
FROM employee e;

/* 10. Write a SQL query to return the “Attrition”, “Department”, “Gender”, “EmployeeNumber” from the ”hr” database for those employees in sales department.*/
SELECT Attrition, Department, Gender, EmployeeNumber 
FROM employee e
Where Department = "Sales";


/* 11. Write a SQL query to return the “EmployeeNumber”, “Department”, “EducationField”, “MaritalStatus”, “Attrition”, from the ”hr” database 
 * for those employees in the Life Sciences field.*/
SELECT EmployeeNumber, Department, EducationField, MaritalStatus, Attrition 
FROM employee e
WHERE EducationField = "Life Sciences";


/*12. Write a SQL query to return the “EmployeeNumber”, “Department”, “HourlyRate”, “JobRole”, “Attrition”, from the ”hr” database for those employees whose hourly rates are greater than 65.
 * Sort the returned table in a descending order by the field “HourlyRate.”*/
SELECT EmployeeNumber, Department, HourlyRate, JobRole, Attrition 
FROM employee e
WHERE HourlyRate > 65
ORDER BY HourlyRate Desc;

/*13. Write a SQL query to return the “EmployeeNumber”, “JobRole” from the ”hr” database for those employees whose “JobRole” contains the “Technician” keyword.*/
Select EMployeeNumber, JobRole
FROM employee e
WHERE JobRole LIKE "%Technician%";


/*14. Write a SQL query to return the “EmployeeNumber”, “JobRole” from the ”hr” database for those employees whose “JobRole” ends with “Representative”.*/
Select EMployeeNumber, JobRole
FROM employee e
WHERE JobRole LIKE "%Representative";


/* 15. Write a SQL query to return the “EmployeeNumber”, “JobRole” from the ”hr” database for those employees whose “JobRole” has the “Research” as the first word. */
Select EMployeeNumber, JobRole
FROM employee e
WHERE JobRole LIKE "Research%";





