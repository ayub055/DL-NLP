## Declaring Path 
#/data/home/ayyoobmohd/DLNLP/Glove-and-Sentiments/Datasets/ClassificationDataset1.xlsx
pos_path = '/data/home/ayyoobmohd/DLNLP/Glove-and-Sentiments/data/positive_reviews.csv'
neg_path = '/data/home/ayyoobmohd/DLNLP/Glove-and-Sentiments/data/negative_reviews.csv'

def making_new_dataset(data):
    data.to_csv(pos_path, index = False, header= False,
          encoding = "latin-1", columns = ['Positive Review'])
    
    data.to_csv(neg_path, index = False, header= False,
          encoding = "latin-1", columns = ['Negative Review'])
    
    
    positive_set = open(pos_path, "r", encoding="latin-1").read()
    negative_set = open(neg_path, "r", encoding="latin-1").read()
    
    pos_set = positive_set.split("\n")[:-1]
    neg_set = negative_set.split("\n")[:-1]
    
    #print(len(positive_data), len(negative_data))
    
    return pos_set, neg_set

positive_data, negative_data = making_new_dataset(data) 
