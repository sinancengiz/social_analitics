

```python
# Dependencies
import json
import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from datetime import datetime
from pprint import pprint
```


```python
# Import Twitter API Keys
from config import consumer_key, consumer_secret, access_token, access_token_secret
```


```python
# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Hashtags
target_user =["@BBC", "@CBSNews", "@cnni", "@FoxNews", "@nytimes"]
```


```python
# List for dictionaries of results
user_results_means_list= []
user_result_list = []
bbc_results = []
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
#get created at to see the result is coming 
public_tweets = api.user_timeline("@BBC", page=1)
pprint(public_tweets[1]["user"]["created_at"])
```

    'Thu Jan 29 08:30:16 +0000 2009'
    


```python
# Loop through each user
for user in target_user:

    # Variables for holding sentiments
    tweet_text_list = []
    tweet_times = []
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []
    
    # Loop through 5 pages of tweets (total 100 tweets)
    for x in range(1, 6):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)

        # Loop through all tweets
        for tweet in public_tweets:

            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            tweet_text = tweet["text"]
            tweet_time = tweet["user"]["created_at"]
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
           # Add each value to the appropriate list
            tweet_text_list.append(tweet_text)
            tweet_times.append(tweet_time)
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)

    # Create a dictionaty of results
    user_results_means = {
        "Compound Score": np.mean(compound_list),
        "Postive Score": np.mean(positive_list),
        "Neutral Score": np.mean(neutral_list),
        "Negative Score": np.mean(negative_list)
        }
    user_results = {
        "Text" : tweet_text_list,
        "Tweet Time" : tweet_times,
        "Compound Score": compound_list,
        "Postive Score": positive_list,
        "Neutral Score": neutral_list,
        "Negative Score": negative_list
        }
        
    # Append dictionary to list
    
    user_results_means_list.append(user_results_means)
    
    if user == "@BBC":
        bbc_df = pd.DataFrame(user_results)
    elif user == "@CBSNews":
       cbs_df = pd.DataFrame(user_results)
    elif user == "@cnni":
       cnn_df = pd.DataFrame(user_results)
    elif user == "@FoxNews":
       fox_df = pd.DataFrame(user_results)
    elif user == "@nytimes":
       nyt_df = pd.DataFrame(user_results)
```


```python
nyt_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound Score</th>
      <th>Negative Score</th>
      <th>Neutral Score</th>
      <th>Postive Score</th>
      <th>Text</th>
      <th>Tweet Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.6908</td>
      <td>0.207</td>
      <td>0.793</td>
      <td>0.000</td>
      <td>The tiny elevator in a Brooklyn housing projec...</td>
      <td>Fri Mar 02 20:41:42 +0000 2007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>South Carolina's former first lady, Jenny Sanf...</td>
      <td>Fri Mar 02 20:41:42 +0000 2007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.2776</td>
      <td>0.064</td>
      <td>0.823</td>
      <td>0.113</td>
      <td>Modern Love: "I’m not sure it’s possible to ju...</td>
      <td>Fri Mar 02 20:41:42 +0000 2007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.4215</td>
      <td>0.000</td>
      <td>0.797</td>
      <td>0.203</td>
      <td>Review: ‘Mean Girls’ Sets the Perils of Being ...</td>
      <td>Fri Mar 02 20:41:42 +0000 2007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.1027</td>
      <td>0.188</td>
      <td>0.680</td>
      <td>0.132</td>
      <td>Michigan will stop providing free bottled wate...</td>
      <td>Fri Mar 02 20:41:42 +0000 2007</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Add timestamp to dataFrames
bbc_df["Timestamp"] = range(1, 101)
cbs_df["Timestamp"] = range(1, 101)
cnn_df["Timestamp"] = range(1, 101)
fox_df["Timestamp"] = range(1, 101)
nyt_df["Timestamp"] = range(1, 101)
```


```python
#reorganise the dataFrames Collumns
organized_bbc_df = bbc_df[["Timestamp","Tweet Time","Text","Compound Score","Neutral Score","Negative Score","Postive Score"]]
organized_cbs_df = cbs_df[["Timestamp","Tweet Time","Text","Compound Score","Neutral Score","Negative Score","Postive Score"]]
organized_cnn_df = cnn_df[["Timestamp","Tweet Time","Text","Compound Score","Neutral Score","Negative Score","Postive Score"]]
organized_fox_df = fox_df[["Timestamp","Tweet Time","Text","Compound Score","Neutral Score","Negative Score","Postive Score"]]
organized_nyt_df = nyt_df[["Timestamp","Tweet Time","Text","Compound Score","Neutral Score","Negative Score","Postive Score"]]

```


```python
#Get the all compound values and put them in single dataFrame
compound_values_df = pd.DataFrame(organized_bbc_df["Timestamp"])
compound_values_df["BBC_Compound"] = organized_bbc_df["Compound Score"]
compound_values_df["CBS_Compound"] = organized_cbs_df["Compound Score"]
compound_values_df["CNN_Compound"] = organized_cnn_df["Compound Score"]
compound_values_df["FOX_Compound"] = organized_fox_df["Compound Score"]
compound_values_df["NYT_Compound"] = organized_nyt_df["Compound Score"]
#compound_values_df["Timestamp"] = organized_bbc_df["Timestamp"]
compound_values_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Timestamp</th>
      <th>BBC_Compound</th>
      <th>CBS_Compound</th>
      <th>CNN_Compound</th>
      <th>FOX_Compound</th>
      <th>NYT_Compound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.7096</td>
      <td>0.0000</td>
      <td>-0.6908</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.4939</td>
      <td>0.3400</td>
      <td>-0.8834</td>
      <td>-0.2411</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.2500</td>
      <td>-0.8258</td>
      <td>-0.3818</td>
      <td>0.3818</td>
      <td>0.2776</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.4215</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0000</td>
      <td>-0.7506</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.1027</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create a scatter plot which contains all the madia outlets compound values
ax = compound_values_df.plot(kind="scatter", x="Timestamp",y="BBC_Compound", color="black", edgecolors="black", s=60,  alpha=0.80, label="BBC")
compound_values_df.plot(kind="scatter", x="Timestamp",y="CBS_Compound", color="green", edgecolors="black",s=60, alpha=0.80, label="CBS", ax=ax)
compound_values_df.plot(kind="scatter", x="Timestamp",y="CNN_Compound", color="red", edgecolors="black",s=60, alpha=0.80,label="CNN", ax=ax)
compound_values_df.plot(kind="scatter", x="Timestamp",y="FOX_Compound", color="blue", edgecolors="black",s=60, alpha=0.80,label="FOX", ax=ax)
compound_values_df.plot(kind="scatter", x="Timestamp",y="NYT_Compound", color="yellow", edgecolors="black",s=60, alpha=0.80, label="NYT", ax=ax)
#create a title
plt.title("Sentiment Analysis of Media Tweets (04/07/2018)")
#create a y label
plt.ylabel("Tweet Polarity")
#set background color
ax.set_facecolor('xkcd:pale turquoise')
#Putting Legend
plt.legend(title = "Media Outlet", loc="upper right",bbox_to_anchor=(1.25, 1))
#set x and y limits
plt.xlim([105,-5]) #Bonus
plt.ylim([-1.05,1.05]) #Bonus
#set a grid
plt.grid()
#plot the chart
plt.show()
```


![png](output_12_0.png)



```python
#create a dataframe from user results means data
overall_sentiments_df = pd.DataFrame(user_results_means_list)
#put target users in to the data frame
overall_sentiments_df["Media Outlet"] = target_user
#reorganising the collumns
overall_sentiments_df = overall_sentiments_df[["Media Outlet","Compound Score","Neutral Score","Negative Score","Postive Score"]]
overall_sentiments_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Media Outlet</th>
      <th>Compound Score</th>
      <th>Neutral Score</th>
      <th>Negative Score</th>
      <th>Postive Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>@BBC</td>
      <td>0.091746</td>
      <td>0.86137</td>
      <td>0.04773</td>
      <td>0.09089</td>
    </tr>
    <tr>
      <th>1</th>
      <td>@CBSNews</td>
      <td>-0.194165</td>
      <td>0.79305</td>
      <td>0.15203</td>
      <td>0.05489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>@cnni</td>
      <td>-0.080350</td>
      <td>0.82279</td>
      <td>0.10508</td>
      <td>0.07214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>@FoxNews</td>
      <td>-0.101493</td>
      <td>0.82154</td>
      <td>0.11492</td>
      <td>0.06355</td>
    </tr>
    <tr>
      <th>4</th>
      <td>@nytimes</td>
      <td>0.021295</td>
      <td>0.83309</td>
      <td>0.08406</td>
      <td>0.08286</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Creating bar chart that show the madia outlet overal sentiment data
bar_chart = plt.bar(overall_sentiments_df["Media Outlet"],overall_sentiments_df["Compound Score"], 
                    color ='brown', alpha=0.5, align="edge")
x_axis = np.arange(len(overall_sentiments_df["Media Outlet"]))
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, ["BBC", "CBS", "CNN", "FOX", "NYT"])
# add the title
plt.title("Overall Media Sentiment Based on Tweeter (04/07/2018)")
#add labels
plt.ylabel("Tweet Polarity")
plt.xlabel("Media Outlets")
#add color to the bars
bar_chart[1].set_color('green')
bar_chart[2].set_color('red')
bar_chart[3].set_color('blue')
bar_chart[4].set_color('yellow')
#set y limit
plt.ylim([-0.3,0.15]) #Bonus
#add grid
plt.grid()
plt.show()
```


![png](output_14_0.png)



```python
#Excract data to csv
writer = pd.ExcelWriter('Media_Outlet_result.xlsx')
overall_sentiments_df.to_excel(writer, index=False, sheet_name='Overal Media Sentiment')
organized_bbc_df.to_excel(writer,index=False, sheet_name='BBC')
organized_cbs_df.to_excel(writer,index=False, sheet_name='CBS')
organized_cnn_df.to_excel(writer,index=False, sheet_name='CNN')
organized_fox_df.to_excel(writer,index=False, sheet_name='FOX')
organized_nyt_df.to_excel(writer,index=False, sheet_name='NYT')
writer.save()
```
