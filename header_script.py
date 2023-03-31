columns = []

for i in range(784):
    columns.append(f"a{i}")
print(f"letter,{','.join(columns)}")