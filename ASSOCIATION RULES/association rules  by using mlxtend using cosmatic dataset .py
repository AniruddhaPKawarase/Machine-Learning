import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

fp_cos=pd.read_csv("/home/dai/Downloads/ML DATA/Datasets/Cosmetics.csv",index_col=0)


############ Support of 1 item sets
itemFreq=fp_cos.sum(axis=0)
plt.figure(figsize= (8,5))
plt.bar(itemFreq.index,itemFreq,zorder=3,color="orange")
plt.xticks(rotation=90)
plt.grid(True)
plt.show()


####### Create frequent itemsets

itemsets=apriori(fp_cos,min_support=0.2,use_colnames=True,)

# 7     0.536              (Foundation)
# 8     0.490               (Lip Gloss)
# 10    0.457                (Eyeliner)


############# Convert it into rules

rules=association_rules(itemsets,metric='confidence',min_threshold=0.6)
rules

#     antecedents   consequents  ...  leverage  conviction
# 0       (Blush)   (Concealer)  ...  0.059554    1.416462
# 1    (Eyeliner)   (Concealer)  ...  0.095006    1.593787
# 2   (Concealer)    (Eyeliner)  ...  0.095006    1.655214
# 3     (Mascara)  (Eye shadow)  ...  0.184983    6.138417
# 4  (Eye shadow)     (Mascara)  ...  0.184983    4.083050
# 5  (Foundation)   (Lip Gloss)  ...  0.093360    1.518667
# 6   (Lip Gloss)  (Foundation)  ...  0.093360    1.696716





