from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

#Sample dataset 
dataset = [
    ['milk', 'bread', 'nuts', 'apple'],
    ['milk', 'bread', 'nuts'],
    ['milk', 'bread'],
    ['milk', 'apple'],
    ['bread', 'apple'],
    ['milk', 'apple'],
    ['milk', 'bread', 'apple'],
]


te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

#Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)

#Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

#Display results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
