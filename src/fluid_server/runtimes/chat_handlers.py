"""
Custom chat handlers for multimodal models
"""

import logging
from llama_cpp.llama_chat_format import Llava15ChatHandler

logger = logging.getLogger(__name__)


class Gemma3ChatHandler(Llava15ChatHandler):
    """Gemma 3 chat handler with proper <start_of_turn>/<end_of_turn> format"""
    
    # Chat Format:
    # '<bos><start_of_turn>user\n{system_prompt}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n'

    DEFAULT_SYSTEM_MESSAGE = None

    CHAT_FORMAT = (
        "{% if messages[0]['role'] == 'system' %}"
        "{% if messages[0]['content'] is string %}"
        "{% set first_user_prefix = messages[0]['content'] + '\n\n' %}"
        "{% else %}"
        "{% set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' %}"
        "{% endif %}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{% set first_user_prefix = \"\" %}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
        "{{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}"
        "{% endif %}"
        "{% if (message['role'] == 'assistant') %}"
        "{% set role = \"model\" %}"
        "{% else %}"
        "{% set role = message['role'] %}"
        "{% endif %}"
        "{{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] | trim }}"
        "{% elif message['content'] is iterable %}"
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image_url' and item['image_url'] is string %}"
        "{{ '\n\n' + item['image_url'] + '\n\n' }}"
        "{% elif item['type'] == 'image_url' and item['image_url'] is mapping %}"
        "{{ '\n\n' + item['image_url']['url'] + '\n\n' }}"
        "{% elif item['type'] == 'text' %}"
        "{{ item['text'] | trim }}"
        "{% endif %}"
        "{% endfor %}"
        "{% else %}"
        "{{ raise_exception(\"Invalid content type\") }}"
        "{% endif %}"
        "{{ '<end_of_turn>\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\n' }}"
        "{% endif %}"
    )