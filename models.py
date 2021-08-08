from time import time
import transformers

mark = time()
print('loading conversational model ...')
# 51s-150s: facebook/blenderbot_small-90M
# 372s?: microsoft/DialoGPT-small
#       facebook/blenderbot-1B-distill
converse_blenderbot_small_90M = transformers.pipeline('conversational', model='facebook/blenderbot_small-90M')
print((time() - mark), 'seconds blenderbot_small-90M')
mark = time()

print('loading classification model ...')
# 10s-36s: typeform/mobilebert-uncased-mnli
classify = transformers.pipeline('zero-shot-classification', model='typeform/mobilebert-uncased-mnli')
print((time() - mark), 'seconds mobilebert-uncased-mnli')
