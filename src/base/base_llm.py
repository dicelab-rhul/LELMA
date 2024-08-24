from abc import ABC, abstractmethod


class BaseLLM(ABC):
	"""
	Abstract Language Model manager class for managing interactions with language models.
	"""

	def __init__(self):
		"""
		Initialize the Language Model Manager.
		"""
		self.messages = []

	@property
	@abstractmethod
	def save_history(self):
		pass

	@property
	@abstractmethod
	def context(self):
		pass

	@property
	@abstractmethod
	def instruction_tokens(self):
		pass

	@property
	@abstractmethod
	def generated_tokens(self):
		pass

	@abstractmethod
	def clear_token_count(self):
		pass

	@abstractmethod
	def prompt(self, instruction, max_tokens=1024) -> str:
		"""
		Abstract method to prompt the language model with an instruction and return the response.
		
		Args:
			instruction (str): The instruction to prompt the language model with.
			max_tokens (int): Maximum number of tokens to generate in the response.

		Returns:
			str: The response from the language model.
		"""
		pass

	@abstractmethod
	def clear_context(self):
		"""
		Abstract method to clear context of the conversation.
		"""
		pass

	@abstractmethod
	def add_response(self, response):
		"""
		Abstract method to add response to context of the conversation.

		Args:
			response (str): Response to be added to history of the conversation.
		"""
		pass

	@abstractmethod
	def get_name(self):
		"""
		Abstract method to get name of a model.
		"""
		pass
