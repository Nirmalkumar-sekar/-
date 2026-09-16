𝗡𝗔𝗜𝗩𝗘 𝗕𝗔𝗬𝗘𝗦 𝗖𝗟𝗔𝗦𝗦𝗜𝗙𝗜𝗖𝗔𝗧𝗜𝗢𝗡 | 𝗠𝗔𝗖𝗛𝗜𝗡𝗘 𝗟𝗘𝗔𝗥𝗡𝗜𝗡𝗚

𝗣𝗿𝗼𝗯𝗹𝗲𝗺 𝗦𝘁𝗮𝘁𝗲𝗺𝗲𝗻𝘁 :

A messaging system wants to predict whether a message is **Spam** or **Not Spam** based on the words present in the message.

𝗖𝗹𝗮𝘀𝘀 𝗦𝗽𝗮𝗺 → Unwanted promotional or suspicious messages
𝗖𝗹𝗮𝘀𝘀 𝗡𝗼𝘁 𝗦𝗽𝗮𝗺 → Normal messages

𝗠𝗮𝗰𝗵𝗶𝗻𝗲 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 𝗧𝘆𝗽𝗲 :

𝗦𝘂𝗽𝗲𝗿𝘃𝗶𝘀𝗲𝗱 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 → 𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻

𝗧𝗲𝗰𝗵𝗻𝗼𝗹𝗼𝗴𝘆 𝗦𝘁𝗮𝗰𝗸 :

🐍 Python
📊 NumPy
🐼 Pandas

𝗧𝗼𝗽𝗶𝗰𝘀 𝗖𝗼𝘃𝗲𝗿𝗲𝗱 :

• Naive Bayes Classification
• Supervised Learning
• Text Classification
• Prior Probability
• Word Probability
• Conditional Probability
• Laplace Smoothing
• Spam vs Not Spam Classification
• Manual Probability Calculation
• Classification using Probability Scores
• Predicting new messages

𝗛𝗮𝗻𝗱𝘀-𝗼𝗻 𝗜𝗺𝗽𝗹𝗲𝗺𝗲𝗻𝘁𝗮𝘁𝗶𝗼𝗻 :

I implemented a simple **Naive Bayes text classification concept from scratch** using NumPy and Pandas.

1️⃣ 𝗣𝗿𝗶𝗼𝗿 𝗣𝗿𝗼𝗯𝗮𝗯𝗶𝗹𝗶𝘁𝘆

First, I calculated the probability of each class:

• P(Spam)
• P(Not Spam)

These probabilities represent how frequently each class appears in the training dataset.

2️⃣ 𝗪𝗼𝗿𝗱 𝗣𝗿𝗼𝗯𝗮𝗯𝗶𝗹𝗶𝘁𝘆

For a given word, I calculated its probability of appearing in each class.

For example:

P("meeting" | Spam)

P("meeting" | Not Spam)

3️⃣ 𝗟𝗮𝗽𝗹𝗮𝗰𝗲 𝗦𝗺𝗼𝗼𝘁𝗵𝗶𝗻𝗴

I used Laplace Smoothing to avoid zero probability when a word has not appeared in a particular class.

Formula:

P(word | class) = (word count + 1) / (class count + 2)

The +1 helps prevent a probability from becoming zero.

4️⃣ 𝗖𝗹𝗮𝘀𝘀 𝗦𝗰𝗼𝗿𝗲

Finally, I multiplied the word probability with the class probability:

Spam Score = P(word | Spam) × P(Spam)

Not Spam Score = P(word | Not Spam) × P(Not Spam)

The class with the higher score becomes the final prediction.

𝗘𝘅𝗮𝗺𝗽𝗹𝗲 :

For the word **"meeting"**, the model compares:

P("meeting" | Spam)

with

P("meeting" | Not Spam)

Since "meeting" appears in the **Not Spam** messages in our dataset, the model calculates the corresponding probabilities and predicts the class with the higher score.

𝗥𝗲𝗮𝗹-𝘁𝗶𝗺𝗲 𝗨𝘀𝗮𝗴𝗲 :

Naive Bayes can be applied to several real-world classification problems, including:

• Email spam detection
• Sentiment analysis
• Text classification
• News category classification
• Document classification
• Customer feedback classification
• Message filtering

𝗞𝗲𝘆 𝗟𝗲𝗮𝗿𝗻𝗶𝗻𝗴 :

𝗠𝗲𝘀𝘀𝗮𝗴𝗲 → 𝗪𝗼𝗿𝗱𝘀 → 𝗣𝗿𝗶𝗼𝗿 𝗣𝗿𝗼𝗯𝗮𝗯𝗶𝗹𝗶𝘁𝘆 → 𝗪𝗼𝗿𝗱 𝗣𝗿𝗼𝗯𝗮𝗯𝗶𝗹𝗶𝘁𝘆 → 𝗦𝗰𝗼𝗿𝗲 → 𝗖𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗰𝗮𝘁𝗶𝗼𝗻

This hands-on exercise helped me understand how a text classification algorithm can use probabilities to make predictions.

It was especially useful to understand the **mathematical intuition behind Naive Bayes and Laplace Smoothing before using ready-made Machine Learning libraries.**

🎤 𝗗𝗶𝘀𝗰𝘂𝘀𝘀𝗶𝗼𝗻 𝗤𝘂𝗲𝘀𝘁𝗶𝗼𝗻 :

❓ Why do we use Laplace Smoothing in Naive Bayes?

If a word has never appeared in a particular class, its probability can become **0**.

When probabilities are multiplied, a single zero probability can make the entire classification score zero.

Laplace Smoothing adds a small adjustment to prevent this problem and allows the model to handle previously unseen words more effectively.

Continuing my hands-on journey in **Machine Learning, Python, AI and Generative AI.** 🚀

#MachineLearning #NaiveBayes #LaplaceSmoothing #SupervisedLearning #Python #PythonProgramming #NumPy #Pandas #DataScience #ArtificialIntelligence #AI #TextClassification #SpamDetection #NLP #MachineLearningProjects #GenerativeAI #GenAI #TechSkills2026 #TechpandaAcademy
