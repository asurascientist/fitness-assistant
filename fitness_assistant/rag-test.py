#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().system('wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py')


# ## Ingestion

# In[1]:


import minsearch
import pandas as pd
from tqdm.auto import tqdm


# In[2]:


df = pd.read_csv('../data/data.csv')


# In[3]:


documents = df.to_dict(orient='records')


# In[4]:


index = minsearch.Index(
    text_fields=['exercise_name', 'type_of_activity', 'type_of_equipment', 'body_part',
       'type', 'muscle_groups_activated', 'instructions'],
    keyword_fields=['id']
)


# In[5]:


index.fit(documents)


# ## RAG flow

# In[6]:


from openai import OpenAI

client = OpenAI()


# In[7]:


def search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=5
    )
    return results


# In[8]:


prompt_template = """
You're a fitness insrtuctor. Answer the QUESTION based on the CONTEXT from our exercises database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
exercise_name: {exercise_name}
type_of_activity: {type_of_activity}
type_of_equipment: {type_of_equipment}
body_part: {body_part}
type: {type}
muscle_groups_activated: {muscle_groups_activated}
instructions: {instructions}
""".strip()

def build_prompt(query, search_results):
    context = ""
    
    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[50]:


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model = model,
        messages=[{"role":"user", "content": prompt}]
    )

    return response.choices[0].message.content


# In[12]:


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


# In[13]:


query = 'Give me chest workout that i can do in home'
rag(query)


# ## Retrieval evaluation

# In[16]:


df_question = pd.read_csv('../data/ground-truth-retrieval.csv')


# In[17]:


df_question.head()


# In[18]:


ground_truth = df_question.to_dict(orient='records')


# In[19]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[20]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[21]:


def evaluate(ground_truth, search_function):
    relevance_total = []
    
    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        relevance = [d['id']== doc_id for d in results]
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total)
    }


# In[22]:


evaluate(ground_truth,lambda q: minsearch_search(q['question']))


# ## Finding the best parameters

# In[23]:


df_validation = df_question[:100]
df_test = df_question[100:]


# In[24]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf') # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)

        # Evaluate the objective function
        current_score = objective_function(current_params)

        # Update best if current is better
        if current_score > best_score: # Change to > if maximizing
            best_score = current_score
            best_params = current_params

    return best_params, best_score


# In[25]:


gt_val = df_validation.to_dict(orient='records')


# In[26]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[27]:


param_ranges = {
    'exercise_name': (0.0, 3.0),
    'type_of_activity': (0.0, 3.0),
    'type_of_equipment': (0.0, 3.0),
    'body_part': (0.0, 3.0),
    'type': (0.0, 3.0),
    'muscle_groups_activated': (0.0, 3.0),
    'instructions': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[28]:


simple_optimize(param_ranges, objective, n_iterations=20)


# In[29]:


def minsearch_improved(query):
    boost = {
        'exercise_name': 2.508948916824827,
        'type_of_activity': 1.7246027195446567,
        'type_of_equipment': 1.104245672557824,
        'body_part': 0.42502530855175946,
        'type': 0.8161522427247936,
        'muscle_groups_activated': 2.6635935233766475,
        'instructions': 0.27645179511525364
    }

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[30]:


evaluate(ground_truth, lambda q: minsearch_improved(q['question']))


# ## RAG evaluation

# In[31]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[32]:


len(ground_truth)


# In[33]:


record = ground_truth[0]


# In[34]:


answer_llm = rag(record['question'])


# In[35]:


print(answer_llm)


# In[37]:


prompt = prompt2_template.format(question=record['question'], answer_llm=answer_llm)
print(prompt)


# In[38]:


import json


# In[39]:


df_sample = df_question.sample(n=200, random_state=1)


# In[40]:


sample = df_sample.to_dict(orient='records')


# In[51]:


def rag(query, model='gpt-4o-mini'):
    search_results = minsearch_improved(query)
    prompt = build_prompt(query, search_results)
    #print(prompt)
    answer = llm(prompt, model=model)
    return answer


# In[52]:


evaluations = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question)

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[53]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[54]:


df_eval.relevance.value_counts(normalize=True)


# In[55]:


df_eval.to_csv('../data/rag-eval-gpt-4o-mini.csv', index=False)


# In[56]:


df_eval[df_eval.relevance == 'NON_RELEVANT']


# In[ ]:


evaluations_gpt4o = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question, model='gpt-4o')

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[ ]:


df_eval = pd.DataFrame(evaluations_gpt4o, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[58]:


df_eval.relevance.value_counts(normalize=True)


# In[ ]:


df_eval.to_csv('../data/rag-eval-gpt-4o.csv', index=False)

