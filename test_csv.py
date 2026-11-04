import csv

file_path = "knowledge/loans/ready_bank_loans.csv"

with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    print("CSV headers:", reader.fieldnames)

    for row in reader:
        print("CSV row:", row)