# TELEGRAM
ADMIN_ID = 62408647               # тот у кого будет доступ к админ-командам
API_ID = 15457256
API_HASH = 'dc9b2b9395d7d445f372358246b5be0f'
PHONE = "79959059938"


# OpenAI
openai_key="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"

# OpenAI parameters
ANSWER_RATIO = 0.95  # отношение максимальной длины ответа сетки к длине вопроса. 0.4 значит что 60% объёма на вопрос, 40% - на ответ
MAX_REQUEST_LENGHT = 4095   # сколько токенов максимально на весь запрос-ответ. максимум - 4096. можно экспериментировать.
CONTIGENCY = 200   # запас токенов на каждый запрос, на всякий случай
TEMPERATURE = 0.5 # температура запроса. читать мануалы чтобы понять что это


EMOJI_REACTION_LIST = ['👍', '👎', '❤️', '🔥',  '🎉', '🤩', '😱', '😁', '😢', '💩', '🤮', '🥰', '🤯', '🤔', '🤬', '👏']

PROMPT_TO_CHECK_EMOTION = f"Define the best emoji to the following text, answer with 1-3 emojis, coma-separated. List of possible emois to use is: {EMOJI_REACTION_LIST}"