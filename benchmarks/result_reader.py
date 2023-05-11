import fileinput

resutls = {}

current_dataset = None
current_data = []

for line in fileinput.input():
    line = line.strip()
    
    if line == '':
        continue

    if 'Precompilation done.' in line:
        continue

    if current_dataset == None:
        current_dataset = line
        continue

    if len(current_data) == 6:
        resutls[current_dataset] = current_data
        current_dataset = line
        current_data = []
        continue

    current_data.append(float(line.split()[0]))

print('name, filter_datasets, load_data, create_model, fit, train, accuracy')
for name, v in sorted(list(resutls.items()), key=lambda x: x[0]):
    # d = train, e = test, x = accuracy
    a, b, c, d, e, x = v
    print(f'{name}, {a}, {b}, {c}, {d}, {e}, {x}')
