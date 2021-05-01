# SQL and Data Science
+ What is SQL
  + main ops
    + read/retureve data
    + write data - add data to a table
    + update data - insert new data  
  + Concepts
    + Database\
      container to store orginazed data; set of related info
    + Tables\
      structured list of data or items.
    + Cols and rows
      + Col:single field in a table
      + row: record in a table
  + Evolution
    + Data modeling
      + orgnazie and structures info into tables
      + rep business process or show relationships
      + types
        + for prediction
        + data tables
      + NoSQL - Not Only SQL
        + storage and retrieval of unstructered data modeld by means other than tab realtions
  + Data Models
    + relational 
      + allow for easy querying and data manipulation
    + transactional
      + opearational database
    + entity
      + distinguishable uniqe and
    + blocks
      + one-to-many
        + customer to invoices
      + many-to-many
        + student to classes
      + one-to-one
        + manager to store
    + ER Diagrams
      + ER model
      + primary key
        + column(uniquely idnetify every row)
      + foreign keys
        + one or more columns indeitfying a single row in another table
      + Chen Notation
        ![img](imgs/chen-not.png)
      + Crow's Foot Notation
        ![img](imgs/crow-not.png)
      + UML Class Diagram Notation
        ![img](imgs/UML-not.png)
+ Retriving Data
  + SELECT
    + ``` 
      SELECT prod_name
      FROM Products;
      ```
    + ```
      SELECT prod_name,prod_id,prod_price
      FROM Products;
      ```

      ```
      SELECT prod_name,
             prod_id,
             prod_price
      FROM Products;
      ```
    + ```
       SELECT *
       FROM Products;
      ```
    + ``` 
       SELECT desired columns
       FROM specific table
       LIMIT number of records;
       ```
      Oracle\
      ```
      SELECT prod_name
      FROM Products
      WHERE ROWNUM <=5;
      ```
      DB2\
      ```
      SELECT prod_name
      FROM Products
      FETCH FIRST 5 ROWS ONLY;
      ```
+ Creating Tables
  ``` 
  CREATE TABLE shoes(
  Id char(10) RIMARY KEY,
  Brand char(10) NOT NULL,
  Price decimal(8,2) NOTNULL,
  Desc Varchar(750) NOT NULL,
  );
  ```
    + every col is either NULL or NOT NULL
    + Primary keys can not be null
    + add Data
    + ```INSERT```
    ```
    INSERT INTO Shoes
    VALUES ('14535974',
             'GUcci',
             '695.00',
             NULL
            );
    ```
    or 
    INSERT INTO Shoes
           (Id,Brand,Price,Desc)
    VALUES ('14535974',
             'GUcci',
             '695.00',
             NULL
            );
    ```
  + Temporary Tables
    + deleted when cur session is terminated
    + faster
    + useful to complex queries for subset and joins
    ```
    CREATE TEMPORARY TABLE Sandals AS(
        SELECT *
        FROM shoes
        WHERE shoe_type = 'sandals'
    )
    ```
+ Adding Comments
  + multi-lines:```/* */```
  + single"```- -```