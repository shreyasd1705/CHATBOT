## It carries the dataset file. Use the following code to generate the dataframe
fp = open("Sarcasm_tweets.txt", 'r')
id_tweet_map = {}
tweet_id_map = {}
count = 0
for line in fp:
  line = line.strip()
  tokens = line.split(' ')
  if len(tokens) == 1 and tokens[0] != '':
    current_id = tokens[0]
  elif len(tokens) > 1:
    id_tweet_map[current_id] = line

import string
fp = open("Sarcasm_tweet_truth.txt", 'r')
id_truth_map = {}

for line in fp:
  line = line.strip()
  if line == '':
    continue
  elif line[0] in string.digits:
    current_id = line
  else:
    id_truth_map[current_id] = line

id_tweet = pd.DataFrame(id_tweet_map.items(), columns=['tweet_id', 'tweet'])
lbl_tweet = pd.DataFrame(id_truth_map.items(), columns=['tweet_id', 'sarcasm'])
dataset = pd.merge(id_tweet, lbl_tweet, on='tweet_id')
dataset = dataset.replace({'sarcasm': {'YES': 1, 'NO': 0}})
