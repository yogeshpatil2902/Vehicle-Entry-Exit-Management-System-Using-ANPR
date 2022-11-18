#------ IMPORTING THE NECESSARY LIBRARIES ----------#
import mysql.connector
import sys

# ------- PROCESS QUERIES ------#
'''to create a table in database
mycourser.execute("""CREATE TABLE anpr (
                    EMP_ID INT PRIMARY KEY,
                    NAME  VARCHAR(250) NOT NULL,
                    PHONE_NO INT NOT NULL,
                    VECHICLE_NO VARCHAR(250)
                    )""")


to delete any row from table
mycourser.execute("ALTER TABLE my_test.anpr DROP id")

    
db.close() #closing the connection'''


class dbhelper:
    def __init__(self):
        
        try:
            self.conn=mysql.connector.connect(host="localhost", user="root", database="parking",  passwd = "Yogiraj@1998") # alway check this one from line 13 
            self.mycursor = self.conn.cursor()
        
        except:
            print("Not able to connect to database")
            sys.exit(0) # 0 - exit 
            
        else:
            print("Connection done")
            
# ------------ REGISTER THE NEW EMPLOYEE ----------#

    def register(self,sr_no,emp_id,name,phone_no,licence_plate_no):
        try:
            self.mycursor.execute("INSERT INTO employee_data VALUES(%s,%s, %s,%s,%s)",(sr_no,emp_id,name,phone_no,licence_plate_no))
            self.conn.commit() 
        
        except:
            return -1
        else:
            return 1
        
# ------------- FETCHING A PARTICULAR RECORD FROM THE DATABASE

    def search(self,reg_plate):  ## to read the record from DB 
        
        try:
            self.mycursor.execute("SELECT * FROM employee_data WHERE Licence_Plate_No = (%s)",(reg_plate,))
        
            data=self.mycursor.fetchall()
            for x in data:
                print(x)
        
        finally:
            self.conn.close() # disconnecting from server

            