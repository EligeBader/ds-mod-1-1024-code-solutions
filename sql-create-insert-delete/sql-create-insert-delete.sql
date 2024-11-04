CREATE TABLE person_Elige
	(person_id SMALLINT,
	first_name VARCHAR(20),
	last_name VARCHAR(20),
	city VARCHAR(20),
CONSTRAINT pk_person_Elige PRIMARY KEY(person_id));

SELECT * FROM person_Elige;

INSERT INTO person_Elige
		(person_id, first_name, last_name, city)
VALUES
		(1, "Elige", "Bader", "Lake Forest");


INSERT INTO person_Elige
		(person_id, first_name, last_name, city)
VALUES
		(2, "John", "Smith", "Irvine"),
		(3, "Steve", "Platt", "Long Beach");


ALTER TABLE person_Elige
	ADD gender VARCHAR (20); 

UPDATE person_Elige
SET 
	gender = "female"
WHERE 
	person_id = 1;

UPDATE person_Elige
SET 
	gender = "male"
WHERE 
	person_id IN (2,3);

ALTER TABLE person_Elige
DROP COLUMN gender;

DELETE FROM person_Elige
where person_id = 2;

Select * FROM person_Elige;

DROP TABLE person_Elige;