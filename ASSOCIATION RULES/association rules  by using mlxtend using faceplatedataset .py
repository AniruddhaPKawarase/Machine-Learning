import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

fp_df=pd.read_csv("/home/dai/Downloads/ML DATA/Datasets/Faceplate.csv",index_col=0)


############ Support of 1 item sets
itemFreq=fp_df.sum(axis=0)
plt.bar(itemFreq.index,itemFreq)
plt.show()


####### Create frequent itemsets

itemsets=apriori(fp_df,min_support=0.2,use_colnames=True)

############# Convert it into rules

rules=association_rules(itemsets,metric='confidence',min_threshold=0.6)
rules








