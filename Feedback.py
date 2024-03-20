import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score

df = pd.read_csv("ecommerce.csv")

# Drop rows with missing values in the 'Review Text' column
df = df.dropna(subset=['Review Text'])

# Perform sentiment analysis
sid = SentimentIntensityAnalyzer()
df['Sentiment Score'] = df['Review Text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df['Sentiment'] = df['Sentiment Score'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

# Visualize sentiment trends over "time" (order of reviews)
df.reset_index(drop=True, inplace=True)  # Reset index to represent the order of reviews
sentiment_over_time = df['Sentiment'].value_counts().fillna(0)
sentiment_over_time.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# Classify reviews based on 'Class Name'
class_sentiment = df.groupby('Class Name')['Sentiment'].value_counts().unstack().fillna(0)

# Plot pie chart for distribution of positive reviews by class name
positive_reviews = class_sentiment['Positive']
plt.figure(figsize=(8, 6))
plt.pie(positive_reviews, labels=positive_reviews.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Positive Reviews by Class Name')
plt.show()

# Plot pie chart for distribution of negative reviews by class name
negative_reviews = class_sentiment['Negative']
plt.figure(figsize=(8, 6))
plt.pie(negative_reviews, labels=negative_reviews.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Negative Reviews by Class Name')
plt.show()

# Identify which class got the most positive reviews
most_positive_class = positive_reviews.idxmax()
print("Class with most positive reviews:", most_positive_class)



# Generate a word cloud of most frequent words in reviews
text = ' '.join(df['Review Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Reviews')
plt.axis('off')
plt.show()


# Analyze insights and generate recommendations
# Product Quality Improvement
negative_reviews = df[df['Sentiment'] == 'Negative']
product_quality_issues = negative_reviews['Review Text'].str.lower().str.contains('flaw|defect|poor quality|issue|problem|disappointed')
common_product_issues = negative_reviews[product_quality_issues]['Review Text']

# Customer Service Enhancement
customer_service_feedback = df[df['Sentiment'] == 'Negative']
improve_customer_service = customer_service_feedback['Review Text'].str.lower().str.contains('customer service|support|help|response')
common_customer_service_issues = customer_service_feedback[improve_customer_service]['Review Text']

# User Experience Optimization
user_experience_feedback = df[df['Sentiment'] == 'Negative']
improve_user_experience = user_experience_feedback['Review Text'].str.lower().str.contains('user experience|website|app|navigation|usability')
common_user_experience_issues = user_experience_feedback[improve_user_experience]['Review Text']

# Overall Recommendations
print("Product Quality Improvement:")
print("Common Product Quality Issues:")
print(common_product_issues)
print("\nCustomer Service Enhancement:")
print("Common Customer Service Issues:")
print(common_customer_service_issues)
print("\nUser Experience Optimization:")
print("Common User Experience Issues:")
print(common_user_experience_issues)
print("\nOverall Recommendations:")
print("- Regularly monitor customer feedback and sentiment analysis results.")
print("- Prioritize addressing common issues identified in product quality, customer service, and user experience.")
print("- Implement iterative improvements based on customer feedback to enhance overall satisfaction.")


df['Sentiment Score'] = df['Review Text'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
df['Predicted Sentiment'] = df['Sentiment Score'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

# Generate synthetic labels (for demonstration purposes)
df['Actual Sentiment'] = ['Positive' if i % 2 == 0 else 'Negative' for i in range(len(df))]

# Calculate accuracy
accuracy = accuracy_score(df['Actual Sentiment'], df['Predicted Sentiment'])

print("Accuracy of the sentiment analysis model:", accuracy)