import mysql.connector as connector
con=connector.connect(host='localhost’, port=’3306’, user=’root’, password='12345678’, database='scalarAcademy')
query="create table if not exists enrollment(userId int primary key, userName varchar(50), phone varchar (20), DOB varchar(50), emailaddress varchar(50), fatherName varchar(50), mother Name varchar(50), address varchar(50))’’
cur=con.cursor()
cur.execute(query)
print()
def new table():
        query=’create table if not exists courses(introduction_to_programming  varchar(10), AdvancedDSA varchar(10), Concurrent_programming varchar(10), Product_Management varchar(10), problem_solving varchar(10))'
cur=con.cursor()
cur.execute(query)
print(query)

#new_table()
def  upcoming_events():
          query 'create table if not exists upcoming events (Hackthon varchar(10), Coding ninjas varchar(10), Cultural_fest varchar(10), Annual Competitions varchar(10), Special Events varchar(10), Recruitor Summit varchar(10))"
cur=con.cursor()
cur.execute(query)
print(query)

def insert_user(userld, userName, phone, DOB, emailaddress, father Name, motherName, address):
         query="insert into enrollment (userld, userName, phone, DOB, emailaddress, father Name, mother Name, address)values(‘{}’, ‘{}’,‘{}’,‘{}’,‘{}’,‘{}’,‘{}’,‘{}’,)’’.format(userld, userName, phone, DOB, emailaddress, fatherName, motherName, address)
print(query)
cur=con.cursor()
cur.execute(query)
con.commit()
print(‘’user saved to db")
# fetching of data
def fetch_all():
      query= "select * from enrollment’’
     print()
    cur = con.cursor()
   cur.execute(query)
  for row in cur:
        print("UserID:", row[0])
       print("userName:", row[1])
      print("phone:", row[2])
     print("DOB:", row[3])
    print("emailaddress:", row[4])
   print("fathername:", row[5])
  print("fathername:", row[5])
 print("mothername:", row[6])
 print("address:", row[7])
print()
print()

def fetch_one(userId):
      query = "select * from enrollment where userld ={}".format(userId)
      print()
     cur=con.cursor()
     cur.execute(query)
     for row in cur:
           print("UserID: ", row[0])
          print("userName:", row[1])
          print("phone:", row[2])
         print("DOB:", row[3])
         print("emailaddress:", row[4])
        print("fathername:", row[5])
       print("mothername: ", row[6])
      print("address:", row[7])
     print()
def delete_user(userId):
         query="delete from enrollment where userId= {}".format(userId)
         print(query)
        cur = con.cursor()
       cur.execute(query)
      con.commit()
      print("deleted")

def update_user(userId, newName, newphone):
          query="update enrollment set userName='()' phone='{}' where userid=()".format(newName, newphone, userld)
         print(query)
         cur = con.cursor()
        cur.execute(query)
        con.commit()
       print("updated")

def main():
        while True:
            try:
              print(‘’***************************WELCOME TO SCALER  
ACADEMY***************************\n’’)
              print("Choose From The Below Menu")
            print("1. Student Information\n")
            print("2. Courses\n")
           print("3. Upcoming Events\n")
          print("4. Enrollment\n")
         print("5. Record\n")
        print("6. Hostel Information\n")
       print("7. Exit\n")
      ch = int(input())
      if ch==1:
         print("")


      stu_info()
elif ch == 2:
     print("")
    courses()
elif ch == 3:
     print("")
    upcoming_events()
elif ch == 4:
     print("")
    enrollment()
elif ch == 5:
     print("")
    record()
elif ch == 6:
     print("")
    exit()
else:
     print("Invalid Input | Try Again\n")
except Exception as e:
       print(e)
      print("Invalld Details | Try Again\n")
def courses():
      try:
         print("********SELECT COURSES*********)
        print("-----------------------------‘’)
       print("1. Introduction to Programming\n")
       print("2. Advanced DSA\n")
      print("3. Concurrent Programming\n")
     print("4. Product Management\n")
    print("S. Problem Solving & CS Fundamentals\n")
    print("6. Back to Home Page\n")
   ch= int(input())


if ch ==1:
    print("you selected Introduction To programming") e = int(input("press 0 to return to home page\t"))

if e ==0

main()

else:

print("Invalid Value | Try again\n") print()

elif ch == 2:

print("you selected Advanced DSA") a = int(input("press 0 to return to home page\t"))

If a ==0

main()

else:

print("Invalid Value | Try again\n")

print()
