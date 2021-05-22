# Project Description 
The purpose of this project is to determine whether or not Latent Dirichlet Allocations can effectively be used to determine the topical focus of user sentiment from short text social media content. Using user data from Nextdoor.com, I wanted to evaluate the emotional sentiment of posts pertaining to homelessness in the east Denver Area. 

# Executive Summary
- Low Alpha and Beta Parameters
    - Short text and social media origins mean fewer topics per doc and words per topic
- Visualizing Model Training with Gradient Stylesheet
    - Model training is difficult to visualize but we think we found an effective way
- Chunk-size has importance when training on short text
- Simple Sentiment Analysis leaves room for improvement 
    - Future iterations to utilize LSTM's & lda2vec word vectors

## Results: 
By applying a special set of preprocessing steps including part of speech specific word tokens and lemmatization, Latent Dirichlet Allocations can be used to effectively filter short text social media content. Creating a labeled data set using Latent Dirichlet Allocations in order to train a binary classifier would be a sound tactic to employ in further data collection of homelessness content.



# NextDoor 
___
Nextdoor.com is a social platform that required users to register and prove their address with registered mail. This creates a different kind of social network environment from facebook, twitter, and others. Next door claims to strive to create an online space for real world communitied to connect and share. User addresses are used to determine a users Neighborhood as defined through a districting system mutualy governed by senior nextdoor users, and nextdoor staff. Users feed's are comprised of posts from members of ones own , or adjacent neighborhoods. While you can befriend other users, it is not necisarry, and the primary use of the platform is by far as a forum. Nextdoor has no public API and no accessible way to querry user posts beyond ones imediate neighbourhood. 

While Nextdoor markets themselves as community focused, exploring the sites content left myself and others among the Denver Non-profit collective wondering at the braziness and consistency with which discussions on topics such as race, inequality, the diadvantaged , and other public health and social welfare concerns, took polar and propagandized turns. It would appear that the relative isolation and protection of the Nextdoor online community may be specifically enticing to certain types of users, or at least affords users an increased sense of security which appears to noticibaly change the tone/content of user discussions.

# Data Collection 
___
We took two approached to overcoming the geographicly restricted view of user content.

    1.) Implemented volunteer user accounts in data collection 
 
    2.) Contacted Volunteer for cooperation with project

‘Neighbors Helping Neighbors’ - a partnership between Nextdoor and Walmart intended to help communities assist one another throughout the Covid Pandemic with errands or shopping trips. This pesky deal seems to mean that Nextdoor is compelled to share post about Walmart regardless of the term that you query. As such we will likely need to remove ALL posts pertaining to Walmart. This is not an ideal situation as their are examples of when the content we want includes mention of a Walmart in one way or another, but these are the limitations of working without an API.

I was able to scrape over 1,000 posts and +15,000 comments.

# Topic Modeling 
___

The Gensim library was utilized for this project. Sci-Kit learn has comparable tool's, but working with Gensim allows us to use an id2word dictionary rather than a sparse matrix. This dictionary essentially maps each word in the corpus to a unicode value in memory , preventing the creation of duplicates and freeing up a lot of memory. The json file format typically used to store id2word dictionaries is also much lighter on memory and speeds up text processing.

## Data Structure 


Consider a situation in which the comments of a social media post represent topical content quiet different from that of the original post. This is often the case and I'm sure an example wasn't hard to imagine. For this reason we wanted to conduct our topic modeling using the the entirety of the text from both original post and comments.
 

## Chunksize 
One of my more interesting findings throughout this project is that the Chunksize used to train an LDA model can have a drastic effect on its output. It can not be said from this work whether the same would apply when topic modeling for longer texts, but a smaller chunksize when training appears to be the best way to inform the model when working with short text. The loss of human interpretability may be due to the fact that our posts containing disparate user vocabularies, or more likely disparate topical content, can force our model to associate the unassociated.

## Alpha & Beta

We began the project with the intuition that a low beta and alpha parameter ( few topics per doc & few words per topic ) would provide the best results when modeling social media short text. Following this intuition I tuned from alpha and Beta parameters below .5 and in the end our model performed well on this assumption. 

## N_topics 

Due to the semi-supervised scrape with selenium we did not need to assume that the topical content of the data we collected was disparate. Their were number of topics we expected to see ( Housing, Encampments, Crime, Food Banks , etc ) and the model would still need to account for the irrelevant content provided by nextdoor's subpar query tool. The assumption of a larger number of topics helped to reduce the parameter combinations to train, and also proved a useful assumption in the end.


## Sentiment Analysis 

The simple sentiment analysis used in this project is meant to provide an example of the output that would be generated by this pipeline once completed. Simple Sentiment is not usually sufficient for adequate analysis without extensive domain specific dictionaries. I used text2emotions and Vadar from simple sentiment analysis. 

Future iterations of this project will employ LSTM and Convolutional Neural networks for sentiment analysis.

# Visualizing Model Training 
Developed two unique methods of visualizing model tuning performance.
- poly shape / size / color chart 
- gradient stylized tabular data 

See Assets.


# Sentiment Analysis 
I chose to use Vadar for simple polarity analysis. Though the applications of this are limited, they are more useful than the current emotional metrics. Using polarity we can target neighborhoods or individuals that appear polarized around issues of concern.


Text2emotions for simple emotional sentiment analysis of social media text is a small python package authored by shivam sharma. The package will be modified with NRC emotional sentiment data and supplementary labeled datasets in future iterations. The outputs provided by the unmodified package were not entirely accurate , but were impressive for a general use sentiment tool. 

The package contains just under 9,000 terms with labels of Sad, Happy, Angry, Fear , and Surprise 


# Issues 
- GeoMapping
    - Nextdoor.com neighborhoods do not align with postal codes and will require additional time and tools to 
        to map accurately 

# Tools 
-Selenium 
-Gensim 
-pyLDAvis
-Spacy en_core_web_lg
-NLTK

# Conclusions
Latent Dirichlet Allocations pass the test and are found to be an effective addition in sentiment analysis pipelines for the purpose of determining the emotional focus of user posts. 

The size of our dataset prevented us from being able to use custom word embeddings in this iteration. Using the Spacy web_core_lg pretrained language model provided us with some, but not as much accuracy as we would have liked in regards to domain specificity. The LDA model still did very well at recognizing the topical content  of our short text even without the addition of the Gibbs Sampling Method or custom word embeddings both of which are in line for application in future iterations. The size of the dataset here created limitations such that our most immediate change to this project in future iterations would be to collate comment data to corresponding post ID’s before Topic Modeling, or conversely obtaining about 3x the volume of short post’s collected in this study. 

The difficulties presented by scraping nextdoor.com are well worth the insight into local community sentiment.

## Data Dictionary 

All_posts.csv

|Column|Description|
|---|---|
| post_id | identifier from nextdoor url used to remove duplicates and collate comments to posts|
| author |user first and last name that suthored the post |
| date | date the post was published |
| post | in posts_df, body of raw text from each user post |
| comments | merged comment data scrapped from post | 
|lemma | lemmatized Noun's Adjectives Verbs and Adverbs of original post |
|all_text | both posts and comments merged into one body of text |

Results csv's

|Columns| Description|
|---|---|
|  |  |
|  |  |
|  |  |
|  |  |