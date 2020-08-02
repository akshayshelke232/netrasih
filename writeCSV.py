import csv

with open('csv1.csv','w',newline='') as f:
    thewriter = csv.writer(f)

    thewriter.writerow(['12:22:','Desk number x cheating'])
    thewriter.writerow(['12:47:','Desk number x cheating'])
    thewriter.writerow(['01:12:','Desk number x cheating'])
    thewriter.writerow(['01:57:','Desk number x cheating'])
    thewriter.writerow(['02:14:','Desk number x cheating'])
    thewriter.writerow(['02:49:','Desk number x cheating'])