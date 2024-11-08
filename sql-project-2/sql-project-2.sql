
use loandb;

/* 1. Create a temp table to store the row quantity of each table in “loandb” and the temp table includes 2 columns,
 * one is “table_name” and the other is “row_quantity.” Show the table in the end. After take a screenshot of the result, then, drop the table.*/

show tables from loandb;


CREATE TEMPORARY TABLE dsstudent.row_quantities
(
  table_name VARCHAR(50),
  table_rows INT
);

INSERT INTO dsstudent.row_quantities (table_name, table_rows)
SELECT 'bureau', COUNT(*) FROM loandb.`bureau`
UNION ALL
SELECT 'bureau_balance', COUNT(*) FROM loandb.bureau_balance
UNION ALL
SELECT 'credit_card_balance', COUNT(*) FROM loandb.credit_card_balance
UNION ALL
SELECT 'installments_payments', COUNT(*) FROM loandb.installments_payments
UNION ALL
SELECT 'previous_application', COUNT(*) FROM loandb.previous_application
UNION ALL
SELECT 'train', COUNT(*) FROM loandb.train
UNION ALL
SELECT 'POS_CASH_balance', COUNT(*) FROM loandb.POS_CASH_balance ;

 
SELECT * FROM dsstudent.row_quantities; 

drop table dsstudent.row_quantities ;

/*Show the monthly and annual income */
Select * from train;

Select 
	AMT_INCOME_TOTAL annual_income, AMT_INCOME_TOTAL/12 monthly_income 
from train;


/* Transform the “DAYS_BIRTH” column by dividing “-365” and round the value to the integer place. Call this column as “age.” */
SELECT ROUND(DAYS_BIRTH/(-365)) age FROM train;
 
/* Show the quantity of each occupation type and sort the quantity in descending orde*/
SELECT
	OCCUPATION_TYPE, COUNT(*) quantity
FROM train
GROUP BY OCCUPATION_TYPE;

/*In the field “DAYS_EMPLOYED”, the maximum value in this field is bad data, can you write a conditional logic to mark these bad data as “bad data”,
 *and other values are “normal data” in a new field called “Flag_for_bad_data”? */
SELECT 
 DAYS_EMPLOYED,
	IF(DAYS_EMPLOYED = (SELECT MAX(DAYS_EMPLOYED) FROM train), 
     'bad data', 
     'normal data') AS Flag_for_bad_data
FROM train;


/* Can you show the minimum and maximum values for both “DAYS_INSTALLMENT” & “DAYS_ENTRY_PAYMENT” fields in the “installment_payments” table 
 * for default v.s. non-default groups of clients?*/
Describe installments_payments;
Describe  credit_card_balance;
Describe  previous_application;
Describe train;


SELECT 
  t.TARGET TARGET,
  MIN(ip.DAYS_INSTALMENT) min_days_installment,
  MAX(ip.DAYS_INSTALMENT) max_days_installment,
  MIN(ip.DAYS_ENTRY_PAYMENT) min_days_entry_payment,
  MAX(ip.DAYS_ENTRY_PAYMENT) max_days_entry_payment
FROM 
  installments_payments ip
INNER JOIN 
  train t ON ip.SK_ID_CURR = t.SK_ID_CURR
GROUP BY 
  t.TARGET;
