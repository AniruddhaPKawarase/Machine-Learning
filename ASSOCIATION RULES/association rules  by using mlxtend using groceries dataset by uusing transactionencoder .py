import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder



Groceries=[]
with open("/home/dai/Downloads/ML DATA/Datasets/Association Rules Datsets-20221125T100924Z-001/Association Rules Datsets/Groceries.csv") as f:Groceries=f.read() 
Groceries=Groceries.split("\n")

Groceries_list=[]
for i in Groceries:
    Groceries_list.append(i.split(","))
print(Groceries_list)  
  
te=TransactionEncoder()
te_ary=te.fit(Groceries_list).transform(Groceries_list)
print(te_ary)

print(te.columns_)

df=pd.DataFrame(te_ary,columns=te.columns_)
print(df)

############ Support of 1 item sets
itemFreq=df.sum(axis=0)
plt.figure(figsize= (25,15))
plt.bar(itemFreq.index,itemFreq,zorder=3,color="orange")
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


####### Create frequent itemsets

itemsets=apriori(df,min_support=0.2,use_colnames=True,)



############# Convert it into rules

rules=association_rules(itemsets,metric='confidence',min_threshold=0.6)
rules

















