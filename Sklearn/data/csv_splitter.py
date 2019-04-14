import csv

with open("", "r") as f:
    csv_reader = csv.reader(f)
    header = csv_reader[0]
    csv1 = header + csv_reader[1:4000]
    csv2 = header + csv_reader[4001:20000]
    csv3 = header + csv_reader[20001:36000]

with open("csv_1.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(csv1)

with open("csv_2.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(csv2)

with open("csv_3.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(csv3)