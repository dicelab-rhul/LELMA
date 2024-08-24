from .prompt_maker import PromptMaker
from src.base.base_llm import BaseLLM
from .solver import Solver
import logging


class LELMA:
	"""
	LELMA class for running reasoning with an LLM and a formal solver.
	"""

	def __init__(self, llm_reasoner, llm_translator, solver, prompt_maker, max_attempts=10, log_file=None, positive_feedback=True):
		"""
		Initialize the LELMA with LLMs, a formal solver, and a PromptMaker object.
		
		Args:
			llm_reasoner (BaseLLM): an LLM instance to reason.
			llm_translator (BaseLLM): an LLM instance to translate natural language reasoning to queries.
			solver (Solver): a Solver instance.
			prompt_maker (PromptMaker): a PromptMaker class instance.
			max_attempts (int): maximum number of attempts of correcting the reasoning
			log_file (str): name of the log file
			positive_feedback (bool): should the feedback be based on the correct values (True) or negation of failed queries (False)
		"""
		self.reasoner = llm_reasoner
		self.translator = llm_translator
		self.solver = solver
		self.prompt_maker = prompt_maker
		self.max_attempts = max_attempts
		self.log_file = log_file
		self.reasoning_log_handler = None
		self.positive_feedback = positive_feedback

	def reasoning_to_queries(self, response, translating_prompt=None, by_paragraph=False) -> list:
		"""
		Translate given paragraph of text to Prolog queries.
		
		Args:
			response (str): Reasoning response of an LLM.
			translating_prompt (str): Prompt for mapping from natural language to queries.
			by_paragraph (bool): Should the translation be performed for each paragraph separately

		Returns:
			list: A list of queries translated from each paragraph.
		"""
		queries = []

		if translating_prompt is None:
			translating_prompt = self.prompt_maker.read_translation_prompt()

		if by_paragraph:
			responses = self.prompt_maker.split_into_paragraphs(response)
		else:
			responses = [response]

		for response in responses:
			mapped_response = self.translator.prompt(translating_prompt + response)
			extracted_queries = self.prompt_maker.extract_translated_queries_from_text(mapped_response)
			if extracted_queries is not None:
				queries += extracted_queries

		return list(set(queries))  # Remove duplicates

	def inverse_translation(self, failed_queries, problem_specific_predicates=None) -> str:
		"""
		Map failed queries to natural language.
		
		Args:
			failed_queries (list): List of failed queries.
			problem_specific_predicates (str): Problem-specific predicates file's path (e.g. a payoff matrix)

		Returns:
			str: Natural language translation of queries.
		"""
		# feedback based on the correct values of variables
		if self.positive_feedback:
			correcting_prompt = self.prompt_maker.fill_feedback_template_positive(failed_queries, self.solver, problem_specific_predicates)
		# feedback based on the negation of failed queries
		else:
			correcting_prompt = self.prompt_maker.fill_feedback_template(failed_queries)
		return correcting_prompt

	def __init_logging(self, reasoning_logger):
		"""
		Initializes logging for reasoning.

		Args:
		- reasoning_logger: The logger object to be configured.
		"""

		# Create a file handler for logging
		reasoning_handler = logging.FileHandler(self.log_file)
		# Set logging level for the handler to INFO
		reasoning_handler.setLevel(logging.INFO)
		self.reasoning_log_handler = reasoning_handler
		# Define the log message format with '~' as separator between log parts
		formatter = logging.Formatter('%(message)s' + '~\n')
		# Set the formatter for the handler
		reasoning_handler.setFormatter(formatter)
		# Add the handler to the logger
		reasoning_logger.addHandler(reasoning_handler)

	def __remove_handler(self, reasoning_logger):
		"""
		Removes file handler.

		Args:
		- reasoning_logger: The logger object to be removed.
		"""
		reasoning_logger.removeHandler(self.reasoning_log_handler)

	def reason(self, problem_specific_predicates, instruction, log_file=None) -> str:
		"""
		The self-correcting reasoning loop.
		
		Args:
			problem_specific_predicates (str): Problem-specific predicates file's path (e.g. a payoff matrix).
			instruction (str): An experimental instruction prompt.
			log_file (str): Name of a log_file
			
		Returns:
			str: LLM's final reasoning.
		"""
		# Clear context before each reasoning loop
		self.reasoner.clear_context()

		# Setup logger
		logging.debug('Debug info')
		self.log_file = log_file
		reasoning_logger = logging.getLogger('Reasoner')
		# Set logging level to INFO
		reasoning_logger.setLevel(logging.INFO)
		if self.log_file is not None:
			self.__init_logging(reasoning_logger)

		# Prepare data
		reasoning = ""
		translating_prompt = self.prompt_maker.fill_translation_template()

		for attempt in range(self.max_attempts):
			# Get natural language reasoning
			reasoning = self.reasoner.prompt(instruction)
			reasoning_logger.info("###ATTEMPT##" + str(attempt) + "~\nRESPONSE##\n" + reasoning)

			# Extract predicates from the reasoning
			extracted_predicates = self.reasoning_to_queries(reasoning, translating_prompt)
			reasoning_logger.info("PREDICATES##\n" + "\n".join(extracted_predicates))

			# If there are extracted predicates, evaluate
			if len(extracted_predicates) > 0:
				failed_queries = self.solver.evaluate_correctness(problem_specific_predicates, extracted_predicates)
				reasoning_logger.info("FAILED QUERIES##\n" + "\n".join(failed_queries))

				if len(failed_queries) == 0:  # all predicates true
					break
				else:
					# Add erroneous reasoning to the conversation context
					self.reasoner.add_response(reasoning)
					instruction = self.inverse_translation(failed_queries, problem_specific_predicates)
					reasoning_logger.info("CORRECTING PROMPT##\n" + str(instruction))

			else:
				break

		self.__remove_handler(reasoning_logger)
		return reasoning
