columns = []

for i in range(784):
    columns.append(f"a{i} float")
print(f"letter,{','.join(columns)}")
