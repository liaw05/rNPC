import  csv


def write_csv(csv_name, content, mul=True, mod="w"):
    with open(csv_name, mod) as myfile:
        mywriter = csv.writer(myfile)
        if mul:
            mywriter.writerows(content)
        else:
            mywriter.writerow(content)