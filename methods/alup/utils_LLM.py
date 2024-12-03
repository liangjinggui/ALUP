

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

from sensitive_constant import API_KEY

MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY

OPENAI_PIRCE = {
   'gpt-3.5-turbo-0125': {'input_token': 0.5 / 1000000, 'output_token': 1.5 / 1000000}, 
   'gpt-4-turbo-2024-04-09': {'input_token': 10.0 / 1000000, 'output_token': 30.0 / 1000000}, 
}

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError,
                                   openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)