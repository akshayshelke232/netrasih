from reportlab.pdfgen import canvas
import csv

with open('csv1.csv',"r") as file:
    reader = csv.reader(file)
    c = canvas.Canvas("genreport.pdf")
    b = 800
    for row in reader:
        p = row[0]
        q = row[1]
        c.drawString(10, b, p)
        c.drawString(60, b, q)
        b-=20
    c.save()







