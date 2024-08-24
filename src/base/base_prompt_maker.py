from abc import ABC, abstractmethod


class BasePromptMaker(ABC):
	@abstractmethod
	def read_prompt_from_file(self, file_path) -> str:
		pass

	@abstractmethod
	def read_translation_prompt(self) -> str:
		pass

	@abstractmethod
	def read_feedback_prompt(self) -> str:
		pass

	@abstractmethod
	def fill_instruction_template(self, instruction_template, input_data) -> str:
		pass

	@abstractmethod
	def fill_translation_template(self) -> str:
		pass

	@abstractmethod
	def extract_translated_queries_from_text(self, text) -> list:
		pass

	@abstractmethod
	def fill_feedback_template(self, input_strings) -> str:
		pass

	@abstractmethod
	def fill_feedback_template_positive(self, input_strings, solver, payoff_matrix_path) -> str:
		pass
