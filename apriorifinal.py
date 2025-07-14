import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

dataset = pd.read_csv(r'basket.csv')

dataset.fillna('1', inplace=True)

Transactions = []

for i in range(14963):
    transaction = []
    for j in range(11):
        if dataset.iloc[i, j] != '1':
            transaction.append(dataset.iloc[i, j])
    Transactions.append(transaction)

te = TransactionEncoder()
te_bin = te.fit_transform(Transactions)
Transactions = pd.DataFrame(te_bin, columns=te.columns_)


def encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


Transactions = Transactions.applymap(encode)

frequent_items = apriori(Transactions, min_support=0.005, use_colnames=True)
print(frequent_items.head())

rules = association_rules(frequent_items, min_threshold=.1)
print(rules.head())

rules = rules.sort_values(by='lift', ascending=False)
print(rules)

# converting to remove frozenset
rules = rules.applymap(lambda x: str(
    list(x))[1:-1] if isinstance(x, frozenset) else x)
print(rules)

# make csv for rules table
ARules = pd.DataFrame(columns=['Rule', 'Support', 'Confidence'])

# rules declare
count = 1
for index, row in rules.iterrows():
    # access the 'name' and 'age' columns using their names
    product1 = row['antecedents']
    product2 = row['consequents']
    support = row['support']
    confidence = row['confidence']

    # do something with the variables
    print(
        f"{count}.  Rule:{product1} -> {product2}\nSupport: {support:.5f}\nConfidence: {format(confidence, '.0%')}\n")
    count += 1

if not os.path.isfile('AssociationRules.csv'):
    ARules.to_csv('AssociationRules.csv', index=False)

# Saving results
filename = 'rules.csv'

# check if the file already exists
if os.path.isfile(filename):
    # if the file exists, rename it with a new name
    i = 1
    while os.path.isfile(f'{filename[:-4]}_{i}.csv'):
        i += 1
    os.rename(filename, f'{filename[:-4]}_{i}.csv')

# write the DataFrame to the CSV file
with open(filename, 'w') as f:
    f.write(rules.to_csv(index=False))
