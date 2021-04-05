# Tool to generate questions and one answer for each question, based on:
# https://github.com/patil-suraj/question_generation
#
# Usage:
# pip3 install -U torch
# pip3 install -U transformers
# pip3 install -U sentencepiece
# pip3 install -U protobuf
# pip3 install -U punkt
#
# python -m nltk.downloader punkt
# python3 test.py

# We cannot use pipeline "question-generation":
# https://github.com/patil-suraj/question_generation/issues/11
#
# Instead, we use pipeline "e2e-qg" to generate questions,
# then use pipeline "multitask-qa-qg" to extract answer for each question.

from pipelines import pipeline
from transformers import pipeline as transformers_pipeline

text = '''
When you apply for citizenship, officials will check your status, verify that you are not prohibited from applying, and ensure that you meet the requirements.

Your application may take several months. Please ensure that the Call Centre always has your correct address while your application is being processed.
'''

print('----------- Original text:')
print(text)

# summarization = transformers_pipeline('summarization', model='google/t5-base')
summarization = transformers_pipeline('summarization')

task_generate_questions = pipeline('e2e-qg', model='valhalla/t5-base-e2e-qg')
task_get_answer = pipeline('multitask-qa-qg', model='valhalla/t5-base-qa-qg-hl')

def get_questions(text):
  return task_generate_questions(text)

def get_answers(questions, text):
  for question in questions:
    answer = task_get_answer({
      'question': question,
      'context': text
    })
    print([{'question': question, 'answer': answer}])

summary = summarization(text, max_length=300, min_length=30, do_sample=False)
print('----------- Summary:')
print(summary)

print('----------- Quizzes:')
sentences = text.split('.')
for sentence in sentences:
  print(sentence)
  questions = get_questions(sentence)
  get_answers(questions, sentence)
  print('')
  print('')
