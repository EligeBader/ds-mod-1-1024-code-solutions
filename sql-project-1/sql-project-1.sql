use dsstudent;

/*1. In the ‘dsstudent’ database, create a permanent table named “customer_{your_name}.*/

CREATE TABLE customer_elige (
							customer_id SMALLINT,
							name varchar(20),
							location varchar(20),
							total_expenditure varchar(20),
							CONSTRAINT customer_id PRIMARY KEY (customer_id)
							);
						
Show columns From customer_elige; 

/*2. Insert the following records to the “customer_{your_name}” table:*/
INSERT INTO 
		customer_elige (customer_id, name, location, total_expenditure)
VALUES 
		(1701, "John", "Newport Beach, CA", "2000"),
 		(1707, "Tracy", "Irvine, CA", "1500"),
 		(1711, "Daniel", "Newport Beach, CA", "2500"),
 		(1703, "Ella", "Santa Ana, CA", "1800"),
 		(1708, "Mel", "Orange, CA", "1700"),
 		(1716, "Steve", "Irvine, CA", "18000");

/*3. Oops! The value in the field ”total_expenditure” of Steve is not correct. It should be “1800.” Can you update this record?*/
UPDATE customer_elige
SET total_expenditure = 1800
WHERE customer_id = 1716;

Select * FROM customer_elige;

/*4. We would like to update our customer data. Can you insert a new column called “gender” in the “customer_{your_name}” table?*/

ALTER TABLE customer_elige
ADD gender varchar(20);

Show columns From customer_elige;

/* 5.Then, update the field “gender” with the following records:*/

UPDATE customer_elige
SET 
	gender = "F"
WHERE 
	customer_id IN (1707, 1703, 1708);

UPDATE customer_elige
SET 
	gender = "M"
WHERE 
	customer_id IN (1701, 1711,1716);

Select * FROM customer_elige;

/* 6. The customer, Steve, decides to quit our membership program, so delete his record from the “customer_{your_name}” table.*/
DELETE FROM customer_elige WHERE customer_id = 1716;

/*7.  Add a new column called “store” in the table “customer_{your_name}”*/
Alter Table customer_elige
ADD store varchar(20);

Show columns From customer_elige;

/* 8. Then, delete the column called “store” in the table “customer_{your_name}” because you accidentally added it.*/ 

/* Sam please let the person who made the excercice know that I should not be hired if I do that much mistakes in my tasks!*/
Alter Table customer_elige
drop store;

/* 9. Use “SELECT” & “FROM” to query the whole table “customer_{your_name}”*/
Select * FROM customer_elige;

/* 10. Return “name” and “total_expenditure” fields from the table “customer_{your_name}”*/
Select name, total_expenditure FROM customer_elige;

/* 11. Return “name” and “total_expenditure” fields from the table “customer_{your_name}” by using column alias (“AS” keyword)*/
Select name n, total_expenditure total_exp FROM customer_elige;

/* 12. Change the datatype of the field “total_expenditure” from “VARCHAR” to ”SMALLINT”*/
Alter Table customer_elige
Modify column total_expenditure SMALLINT;

/* 13. Sort the field “total_expenditure” in descending order*/
Select total_expenditure
FROM customer_elige
ORDER BY total_expenditure DESC;

/* 14. Return the top 3 customer names with the highest expenditure amount from the table “customer_{your_name}”*/
Select name, total_expenditure
FROM customer_elige
ORDER BY total_expenditure DESC
LIMIT 3;

/* 15. Return the number of unique values of the field “location” and use the column alias to name the return field as “nuniques”*/
Select count(DISTINCT(location)) nuniques
FROM customer_elige;

/* 16. Return the unique values of the field “location” and use the column alias to name the return field as “unique_cities”*/
Select DISTINCT(location) unique_cities
FROM customer_elige;

/* 17. Return the data where the gender is male.*/
Select *
FROM customer_elige
WHere gender = "M";

/* 18. Return the data where the gender is female.*/
Select *
FROM customer_elige
WHere gender = "F";

/* 19. Return the data where the location is “Irvine, CA”*/
Select *
FROM customer_elige
WHere location = "Irvine, CA";


/*20. Return “name” and “location” where the ”total_expenditure” is less than 2000 and sort the result by the field “name” in ascending order*/
Select name, location 
FROM customer_elige
Where total_expenditure < 2000 
ORDER BY name ASC;

/* 21. Drop the table “customer_{your_name}” after you finish all the questions.*/

DROP TABLE customer_elige;
 */



