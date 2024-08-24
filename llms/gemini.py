from src.base.base_llm import BaseLLM
import google.generativeai as genai
import os


class Gemini(BaseLLM):
	def __init__(self, save_history=False, temperature=1, model="gemini-pro", context=None):
		"""
		Initialize the Gemini model.

		Args:
			save_history (bool): Should the history of the conversation be retained in subsequent prompts.
			temperature (int): Gemini's temperature parameter value
			model (str): Gemini's model name
			context (str): Content message content
		"""
		super().__init__()
		GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
		genai.configure(api_key=GOOGLE_API_KEY)

		self._save_history = save_history
		self.temperature = temperature
		self.model_name = model
		self.model = genai.GenerativeModel(model_name=model)
		# NOTE: google doesnt make the 3-way distinction: user,system,model. Context has to be set via user message.
		self._context = context
		self.__set_messages()
		self._instruction_tokens = 0
		self._generated_tokens = 0

	@property
	def save_history(self):
		return self._save_history

	@property
	def context(self):
		return self._context

	@property
	def instruction_tokens(self):
		return self._instruction_tokens

	@property
	def generated_tokens(self):
		return self._generated_tokens

	def clear_token_count(self):
		self._instruction_tokens = 0
		self._generated_tokens = 0

	def prompt(self, instruction, max_tokens=1024) -> str:
		"""
		Prompt the language model with an instruction and return the response.

		Args:
			instruction (str): The instruction to prompt the language model with.
			max_tokens (int): Maximum number of tokens to generate in the response.

		Returns:
			str: The response from the language model.
		"""
		self._instruction_tokens += self.model.count_tokens(instruction).total_tokens
		wrapped_message = {"role": "user", "parts": [instruction]}
		# Construct message for prompt
		if not self.save_history:
			self.__set_messages()
		self.messages.append(wrapped_message)

		genai.GenerationConfig(
			max_output_tokens=max_tokens,
			temperature=self.temperature
		)

		# Generate response from Gemiini
		response = self.model.generate_content(self.messages)
		self._generated_tokens += self.model.count_tokens(response.text).total_tokens
		# Extract and return the response content
		return response.text

	def add_response(self, response):
		wrapped_response = {"role": "model", "parts": [response]}
		self.messages.append(wrapped_response)

	def __set_messages(self):
		"""
		Method to handle the context message.
		"""
		if self._context is not None:
			self.messages = [
				{"role": "user",
				 "parts": [self._context]}
			]
		else:
			self.messages = []

	def clear_context(self):
		"""
		Method to clear context of the conversation.
		"""
		self.__set_messages()

	def get_name(self):
		"""
		Method to get name of a model.
		"""
		return self.model_name
