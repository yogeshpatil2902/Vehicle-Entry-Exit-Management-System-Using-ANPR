# -------- IMPORTING NECESSARY LIBRARIES ------------#
import sys
from my_db import dbhelper

# ---------- DISPLAY THE MENU ON TO THE SCREEN -----------#
class vehicle_check:
    def __init__(self):
        #connect to the database
        self.db = dbhelper()
        self.menu()
        
    def menu(self):
        user_input=input("""
                   1.Add the vehical in data base:
                   2.Check if vehicle is already in our database:
                   """)
        if user_input=="1":
            self.register()
                
        elif user_input =="2":
            self.numberplate()
        
        else:
            sys.exit(1000)
            
    # INPUTS FOR REGISTRATION                   
    def register(self):
        sr_no = int(input("Enter Sr.No.::"))
        emp_id = int(input("Enter the Employee ID::"))
        name = input("Enter the Employee Name::")
        phone_no = int(input("Enter the Mobile Number::"))
        licence_plate_no = str(input("Enter the Licence Plate Number::"))
        
        response = self.db.register(emp_id,sr_no,name,phone_no,licence_plate_no)
        
        if response:
            print("Registration successful")
        
        else:
            print("Registration faild")
        
        self.menu()
            
    # INPUT TO SEARCH A PARTICULAR RECORD
    def numberplate(self):
        reg_plate=input("Enter the vehicle's licence plate number::")
        
        self.db.search(reg_plate)


if __name__ =="__main__":
    obj = vehicle_check()
