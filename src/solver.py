from .setup_logger import logger
from swiplserver import PrologMQI


class Solver:
	"""
	Solver class for managing interactions with a solver.
	"""

	def __init__(self, solver_path):
		"""
		Initialize the Solver with the path to the Prolog solver.
		
		Args:
			solver_path (str): Solver path.
		"""
		# Store the path to the Prolog solver
		self.solver = solver_path

	def setup_solver(self, prolog_thread, problem_specific_predicates):
		"""
		Set up the Prolog solver by consulting the Prolog solver file and the payoff matrix file.

		Args:
			prolog_thread: The Prolog thread instance to execute the queries.
			problem_specific_predicates (str): The file path to the problem specific predicates to be consulted by Prolog (e.g. payoff matrix).

		Returns:
			None
		"""
		result = prolog_thread.query("consult(\"" + self.solver + "\").")
		logger.debug("Solver loaded: " + str(result))
		# Add the payoff matrix to the Prolog environment
		problem_specific_predicates = problem_specific_predicates.replace('\\', '/')  # fix for SWIPl paths on Windows
		result = prolog_thread.query("consult(\"" + problem_specific_predicates + "\").")
		logger.debug("Predicates added: " + str(result) + "\n")

	def get_failed(self, queries, prolog_thread):
		"""
		Execute the provided queries and return the ones that failed.

		Args:
			queries (list of str): The list of Prolog queries to be executed.
			prolog_thread: The Prolog thread instance to execute the queries.

		Returns:
			list of str: A list of queries that failed.
		"""
		# Execute the provided queries
		failed_queries = []
		for q in queries:
			logger.debug("query:" + q)
			prolog_thread.query_async(q, find_all=False)

			while True:
				result = prolog_thread.query_async_result()
				if result is None:
					break
				else:
					logger.debug("result:" + str(result) + "\n")
					if not result:
						failed_queries.append(q)
		return failed_queries

	def get_variables_values(self, query, prolog_thread):
		"""
		Execute the provided queries and return their results.

		Args:
			query (str): Prolog query to be executed.
			prolog_thread: The Prolog thread instance to execute the queries.

		Returns:
			list: A list of results from the executed queries.
		"""
		# Execute the provided queries
		final_result = None
		logger.debug("query:" + query)
		prolog_thread.query_async(query, find_all=False)

		while True:
			result = prolog_thread.query_async_result()
			if result is None:
				break
			else:
				logger.debug("result:" + str(result) + "\n")
				final_result = result
		return final_result

	def evaluate_correctness(self, problem_specific_predicates, queries) -> list:
		"""
		Execute queries using the provided payoff matrix.
		
		Args:
			problem_specific_predicates (str): The path to problem-specific predicates (e.g. payoff matrix).
			queries (list): A list of queries to execute.

		Returns:
			list: A list of failed queries.
		"""
		with PrologMQI() as mqi:
			with mqi.create_thread() as prolog_thread:
				# Load the Prolog solver
				self.setup_solver(prolog_thread, problem_specific_predicates)

			failed_queries = self.get_failed(queries, prolog_thread)

		return failed_queries

	def evaluate_value(self, problem_specific_predicates, queries) -> list:
		"""
		Execute queries using the provided payoff matrix and get variables values.

		Args:
			problem_specific_predicates (str): The path to problem-specific predicates (e.g. payoff matrix).
			queries (list): A list of queries to execute.

		Returns:
			list: A list of variables values.
		"""
		with PrologMQI() as mqi:
			with mqi.create_thread() as prolog_thread:
				# Load the Prolog solver
				self.setup_solver(prolog_thread, problem_specific_predicates)

			variables_values = self.get_variables_values(queries, prolog_thread)

		return variables_values
