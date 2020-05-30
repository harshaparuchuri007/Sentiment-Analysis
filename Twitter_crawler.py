import tweepy
import json
from multiprocessing.dummy import Pool as ThreadPool
import time
import datetime

from pymongo import MongoClient

auth = tweepy.OAuthHandler("SVwkLvXBdaEwBxCHTcF83NmDA", "Zv2r1Ze1M8v8qa2YLidrs8yzhhlJw83EP74BLYTSukNp3KrcaN")       #consumer key 
auth.set_access_token("1331394793-nWyRnX8k9uVydR0fhEEIQIVoCI2y9PRRjOQc2T5", "CLq82i56sGHeTDhN2x9ctegHwA130kkIyst8ZWPp1ShPJ") #access token

api = tweepy.API(auth) #providing permission to access data using user twitter keys
trends = ["Democrats","Republicans","democrats","republicans..........etc"]   #provinding the keyword to be searched in the tweets (hashtags)

client = MongoClient()   #creating connection with mongoDB
db = client.term1        #Connection to database
tweet_coll = db.midterm  #Connection to collection of database

def limit_handled(cursor):  #function to handle rate limit error 
    print ("Came here")
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print ("Received Rate Limit Error. Sleeping for 15 minutes , Inside tweets call " + str(datetime.datetime.now()+datetime.timedelta(0,900)))
            time.sleep(15 * 60)
        except tweepy.TweepError:
            print ("Received Rate Limit Error. Sleeping for 15 minutes , Inside tweets call " + str(datetime.datetime.now()+datetime.timedelta(0,900)))
            time.sleep(15 * 60)


def tweets(trend):   #searching and extracting tweets from twitter
    try:
        for status in limit_handled(tweepy.Cursor(api.search, q=trend).items()):
            twt = status.user.location
            if twt != "":
                print (twt +"-->" + trend)
                #print status.entities.get('hashtags')
                status._id = status.id
                del status.id
                tweet_coll.insert_one(json.loads(json.dumps(status._json)))
        print ("Finished querying tweets")
    except tweepy.RateLimitError:
        print ("Received Rate Limit Error. Sleeping for 15 minutes " + str(
            datetime.datetime.now() + datetime.timedelta(0, 900)))
        time.sleep(15 * 60)

if __name__ == '__main__':
    pool = ThreadPool(len(trends))
    pool.map(tweets,trends)

