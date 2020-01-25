# clients_from_hell

scale the features for some of the classifiers




max_pages = 14

for page_number in range(max_pages + 1):
    dynamic_url = f'https://clientsfromhell.net/tag/freelance-felon/page/{page_number}'


Later, the customer's project changed, 
they wanted instead of copying the Clients From Hell site, to instead use the text labelled as coming from the client to classify 
the customers into two lists: Deadbeat or Not. 
Essentially, the customer wanted to answer the question:
    Based on a client's messages to me, do we believe that they will pay?


From our scraped data, we then had to adjust how we stored the data
We are looking specifically for the text attributed to coming from the Client.

We determined that we would use Regular Expressions to review only the sequence of characters between "Client:" and "Me:", to collect the client's messages.

We would still plan to create our dataframe by drawing each Client Type respectively.

from the extracted Client Text - remove punctuation and lowercase the data to standardize it
Then, use Bag-of-words and Term-frequency functions on each observation for each client type.

After our dataframe has been created, we can use a numpy.where function
to identify a Binary Classification in our Target Label. Is the client a Deadbeat or not? 1 or 0?

Separate out just client words, or client+customer, see what we think after testing the models  



Determined that we would focus instead first on the full exchange of the conversation to determine / classify the Client Type as "Yes Deadbeat" or "No Deadbeat"

That way, later once we narrow down the scrape/regex to pull client-side only part of conversation, we can just send that right through the structure we have already built.


Having completed collecting the full conversation data, we concatenated those into a dataframe so that we had each client Type labelled, with individual rows for each observation

Next issue is to determine the Term Frequency - Inverse Document Frequency (TF-IDF) of the words as a whole in our dataframe.
From there, we can determine which columns to drop from our df. Considering we have 15k columns.


Models to select
Multinomial Naive Bayes
    common use case is bag-of-words term frequency
    great results here without having to drop columns at all.

Random Forest, LogisticRegression were clearly overfitted, returning an R-squared score of 1.

Streamlit wordcloud?


Following the TF-IDF, we determined that we would cut out half of the columns, remove the half with the more popular/frequent words

save these as different dataframes and run these dataframes through our models

agreed to all use randomstate=42 for reproducible results



thoughts on the wordcloud -
    for each ClientType, sum the columns for each word, and use these aggregates to populate the wordclouds
    present a wordcloud for each client type? or just "Deadbeat wordcloud" and "nondeadbeat wordcloud" ? 



Streamlit has been moved to a bonus objective - so let's complete most of the work tonight, and have class time tomorrow to work the Bonus

re-scraped data to include only Client Messages
cut down the total data to about half of what it was previously

cont'd with the minmax scaler


next steps 
nearest neighbors (knn)
confusion matrix - how to visualize that, heatmap the confusion matrix? 
throw these into streamlit and do a wordcloud generator