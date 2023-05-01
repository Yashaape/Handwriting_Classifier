# Handwritting Classifer
Manipulate csv file containing data for handwritten letters by importing it into a database
Note: Did not include the line update hw_data_2 set letter = case letter when 0 then 1 else 0 end in sql 
      Used expected_output = 1 if letter == test_letter else 0 in my train_network function
Tested 1,000 random cases for the letter A, B, C, D, and Z.
