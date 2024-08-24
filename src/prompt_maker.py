from src.base.base_prompt_maker import BasePromptMaker
import logging
from .setup_logger import logger
import os
import pandas as pd
import re


### Helper functions ###

def split_into_paragraphs(text) -> list:
	"""
	Split text into paragraphs.

	Args:
		text (str): The text to be split into paragraphs.

	Returns:
		list: A list of paragraphs extracted from the text.
	"""
	return [p.strip() for p in text.split('\n\n') if p.strip()]


def assure_dot(queries):
	"""
	Ensures that each query in the given list ends with a dot.

	Args:
		queries (list of str): A list of queries to be checked.

	Returns:
		list of str: A list of corrected strings where each string ends with a dot.
	"""
	corrected_queries = []
	for query in queries:
		if not query.endswith('.'):
			query = query + '.'
		query = query[:query.index('.') + 1]  # remove everything after the first dot
		corrected_queries.append(query)
	return corrected_queries


def find_matches(input_string, regex):
	"""
	Finds matches in the input string based on the given regex pattern.

	Parameters:
	- input_string (str): Input string to search for matches.
	- regex (str): Regular expression pattern to match against.

	Returns:
	- Tuple: Captured groups if there's a match, otherwise None.
	"""
	match = re.match(regex, input_string)
	if match:
		return match.groups()
	else:
		return None


def is_number(string):
	"""
	Check if the given string represents a valid number (integer or floating point).

	This method uses a regular expression to determine if the string represents a
	number, which can be an integer or a floating-point number, and can optionally
	have a leading negative sign.

	Args:
		string (str): The string to be checked.

	Returns:
		bool: True if the string is a valid number, False otherwise.
	"""
	pattern = re.compile(r'(?<!s)\d+')
	return bool(pattern.match(string))

### End of helper functions ###


class PromptMaker(BasePromptMaker):
	def __init__(self, predicates_tab_path="DATA/TEMPLATES/predicates.csv", templates_dir="DATA/TEMPLATES/",
				 translation_template="translation_prompt.txt", feedback_template="feedback_prompt.txt",
				 prompts_dir="DATA/TEMPLATES/",
				 root="./"):
		"""
		Initializes the PromptMaker class with required parameters.

		Parameters:
		- prompts_tab (str): Path to semicolon-separated csv containing the predicates templates.
		- templates_dir (str): Directory path where template files are stored.
		- translation_template (str): File name of the translation template.
		- feedback_template (str): File name of the feedback template.
		- prompts_dir (str): Directory path where prompts templates are stored.
		- root (str): root directory
		"""
		self.predicates_tab = pd.read_csv(os.path.join(root, predicates_tab_path), sep=";")
		self.templates_dir = os.path.join(root, templates_dir)
		self.translation_template = translation_template
		self.feedback_template = feedback_template
		self.prompts_dir = os.path.join(root, prompts_dir)
		self.root = root

	def read_prompt_from_file(self, file_path) -> str:
		"""
		Read prompt text from a file.
		
		Args:
			file_path (str): The path to the file containing the prompt text.

		Returns:
			str: The prompt text read from the file.
		"""
		prompt = ""
		try:
			with open(file_path, 'r') as file:
				prompt = file.read()
		except FileNotFoundError:
			logging.exception('File ' + file_path + ' does not exist')
		return prompt.strip()

	def read_translation_prompt(self) -> str:
		m_prompt = self.read_prompt_from_file(os.path.join(self.root, self.prompts_dir, self.translation_template))
		return m_prompt

	def read_feedback_prompt(self) -> str:
		c_prompt = self.read_prompt_from_file(os.path.join(self.root, self.prompts_dir, self.feedback_template))
		return c_prompt

	def extract_translated_queries_from_text(self, text) -> list:
		"""
		Extract predicates from text using regular expressions.
		
		Args:
			text (str): The text from which to extract predicates.

		Returns:
			list: A list of predicates extracted from the text.
		"""
		pattern = r"\{(.*?)\}"
		stripped = text.replace("- ", "").replace(" ", "").replace("\t", "").replace("{`", "{").replace("`}",
																										"}").replace(
			"\n", "")  # Remove whitespaces and unnecessary characters
		matches = re.findall(pattern, stripped)
		filtered = [m for m in matches if len(m) > 3]  # Remove final action choice from the list
		syntax_checked = [m for m in filtered if m.count("(") == m.count(
			")")]  # Check if the number of opening and closing parentheses matches
		dot_corrected = assure_dot(syntax_checked)  # Assure each query ends with a dot
		space_formatted = [m.replace(",", ", ") for m in
						   dot_corrected]  # Return to template format with spaces after commas
		result = []
		for query in space_formatted:
			for idx, row in self.predicates_tab.iterrows():
				matches = find_matches(query, row['regex'])
				allowed_str = False
				if matches is not None:
					allowed_str = True
					for match in matches:
						if match.isdigit():
							continue
						elif match in ['B', 'R', 'you', 'them']:
							continue
						else:
							allowed_str = False
				if allowed_str:
					result.append(query)
		return result

	def fill_instruction_template(self, instruction_template, input_data) -> str:
		"""
		Fill an instruction template using the payoff matrix.
		
		Args:
			instruction_template (str): Instruction prompt template.
			input_data (str): File containing data to fill the template (e.g. payoff matrix).

		Returns:
			str: Instruction prompt template filled with data.
		"""
		payoff_matrix = self.convert_pl_to_dict(input_data)
		filled_instruction = instruction_template.format(ul_l=payoff_matrix["UL"][0],
														 ul_r=payoff_matrix["UL"][1],
														 ur_l=payoff_matrix["UR"][0],
														 ur_r=payoff_matrix["UR"][1],
														 dl_l=payoff_matrix["DL"][0],
														 dl_r=payoff_matrix["DL"][1],
														 dr_l=payoff_matrix["DR"][0],
														 dr_r=payoff_matrix["DR"][1], )
		# handle negative payoffs
		filled_instruction = filled_instruction.replace("earn -", "lose ")
		return filled_instruction

	def fill_translation_template(self) -> str:
		"""
		Creates a translation prompt file using the provided mapping template.

		Returns:
		- str: Translation prompt.
		"""
		queries = ""
		short_description = ""

		# Concatenate queries and short descriptions
		for idx, row in self.predicates_tab.iterrows():
			queries += "".join(["- '", row['predicate'], "', where ", row['long_desc'], '.\n'])
			short_description += row['short_desc'] + " or "
		queries = queries[:-2]  # Remove last new line
		short_description = short_description[:-4]  # Remove last or

		# Fill the mapping template with queries and short descriptions
		with open(os.path.join(self.templates_dir, self.translation_template)) as template_file:
			template = template_file.read()
			formatted = template.format(predicates=queries, short_description=short_description)
		return formatted

	def fill_feedback_template(self, input_strings) -> str:
		"""
		Creates a correcting prompt file using the provided correcting template.

		Parameters:
		- input_strings (list): List of input strings to search for matches.

		Returns:
		- str: Correcting prompt.
		"""
		explanation = ""

		# Iterate through input strings and prompts table to find matches and fill templates
		for string in input_strings:
			for idx, row in self.predicates_tab.iterrows():
				matches = find_matches(string, row['regex'])
				if matches is not None:
					template = row['inverse_mapping']
					template = template.format(*matches)

					explanation += "".join(["- ", template, "\n"])
					break

		# Fill the correcting template with the explanation
		with open(os.path.join(self.templates_dir, self.feedback_template)) as template_file:
			template = template_file.read()
			formatted = template.format(explanation=explanation[:-1])  # Remove the last new line
		return formatted

	def fill_feedback_template_positive(self, input_strings, solver, payoff_matrix_path) -> str:
		"""
		Creates a positive (not by negation) correcting prompt file using the provided correcting template.

		Parameters:
		- input_strings (list): List of input strings to search for matches.
		- solver (Solver): solver object
		- payoff_matrix_path (str): path to a payoff matrix

		Returns:
		- str: Correcting prompt.
		"""
		explanation = ""
		variable_names = ['X', 'Y']
		action_names = ['\'B\'', '\'R\'']

		# Iterate through input strings and prompts table to find matches and fill templates
		for string in input_strings:
			for idx, row in self.predicates_tab.iterrows():
				matches = find_matches(string, row['regex'])
				if matches is not None:
					numbers_ind = [index for index, match in enumerate(matches) if is_number(match)]
					correct_values = list(matches)

					query = string
					if len(numbers_ind) > 0:
						if len(numbers_ind) == 2 and len(numbers_ind) == len(matches):
							pass
						else:
							for i, number_ind in enumerate(numbers_ind):
								number_value = matches[number_ind]
								query = query.replace(number_value, variable_names[i], 1)
							values = solver.evaluate_value(payoff_matrix_path, query)

							for ind, value in zip(numbers_ind, values[0].values()):
								correct_values[ind] = value
					else:
						if len(matches) == 1:
							for i, action in enumerate(action_names):
								query = query.replace(action, variable_names[i])

							values = solver.evaluate_value(payoff_matrix_path, query)
							correct_values = list(values[0].values())

						elif "mutual" in query:
							for i, match in enumerate(matches):
								query = query.replace(match, variable_names[i], 1)
							values = solver.evaluate_value(payoff_matrix_path, query.replace('\'', ''))
							correct_values = list(values[0].values())

						else:
							correct_values = correct_values[::-1]

					template = row['inverse_mapping_positive']
					template = template.format(*correct_values)

					explanation += "".join(["- ", template, "\n"])
					break

		# Fill the correcting template with the explanation
		with open(os.path.join(self.templates_dir, self.feedback_template)) as template_file:
			template = template_file.read()
			formatted = template.format(explanation=explanation[:-1])  # Remove the last new line
		return formatted

	def convert_pl_to_dict(self, file_path):
		"""
		Convert a .pl file containing payoff matrix information into a dictionary.

		The .pl file should contain lines in the format:
		payoff('R', 'R', 65, 65).
		payoff('R', 'B', 10, 100).
		payoff('B', 'R', 100, 10).
		payoff('B', 'B', 35, 35).

		The resulting dictionary will have the format:
		{
			"UL": [65, 65],
			"UR": [10, 100],
			"DL": [100, 10],
			"DR": [35, 35]
		}

		Args:
			file_path (str): The path to the .pl file containing the payoff matrix.

		Returns:
			dict: A dictionary with the keys "UL", "UR", "DL", "DR" and corresponding payoff values.
		"""
		# Define a mapping for the combinations to the dictionary keys
		combination_to_key = {
			('R', 'R'): 'UL',
			('R', 'B'): 'UR',
			('B', 'R'): 'DL',
			('B', 'B'): 'DR'
		}

		# Initialize an empty dictionary to store the results
		payoff_dict = {}

		# Open and read the file
		with open(file_path, 'r') as file:
			lines = file.readlines()
			for line in lines:
				# Strip any extra whitespace and remove the period at the end
				line = line.strip().strip('.')
				# Extract the values from the line
				parts = line.split('(')[1].split(')')[0].replace("'", "").split(', ')
				# Get the keys and payoffs
				row_choice, col_choice, payoff1, payoff2 = parts[0], parts[1], int(parts[2]), int(parts[3])
				# Get the dictionary key using the mapping
				key = combination_to_key[(row_choice, col_choice)]
				# Store the payoffs in the dictionary
				payoff_dict[key] = [payoff1, payoff2]

		return payoff_dict


def test(templates_dir, prompts_dir):
	"""
	Test function to demonstrate the usage of PromptMaker class.

	Parameters:
	- templates_dir (str): Directory path where template files are stored.
	- prompts_dir (str): Directory path where prompt files are stored.
	"""
	logging.debug('Test')
	# Read prompts table from CSV file
	prompts_tab = os.path.join(templates_dir, "predicates.csv")
	# Create an instance of PromptMaker
	prompt_maker = PromptMaker(prompts_tab, templates_dir, "mapping.txt", "correcting.txt", prompts_dir, root="../")
	# Generate translation and feedback prompts
	logger.debug(prompt_maker.fill_translation_template())
	prompt_maker.fill_feedback_template(
		["finally(util(you, 10), do(select(you, 'B'), do(select(them, 'R'), s0))).", "safer(B, R)"])
	extracted = prompt_maker.extract_translated_queries_from_text(
		"{finally(util(them, 10), do(choice(you, 'a'), do(choice(them, 'R'), s0))).}"
		"aaa {finally(util(either, 10), do(choice(you, 'B'), do(choice(them, 'R'), s0))).}"
		"{lowest_payoff(10).}"
		"{highest_payoff(10).}"
		"{higher(10, 35).}"
		"{lower(10,35).}"
		"{highest_payoff_for_choice(10, 'R').}"
		"{lowest_payoff_for_choice(10, 'R').}"
		"{highest_payoff(10).}"
		"{lowest_payoff(100).}"
		"{lowest_min_payoff_choice('B')}."
		"{highest_min_payoff_choice('B')}."
		"{highest_possible_payoff(10).}"
		"{lowest_possible_payoff(10).}"
		"{highest_mutual_payoff('B', 'R').}"
		"{lowest_mutual_payoff('B', 'R').}")
	logger.debug(extracted)
	formatted = prompt_maker.fill_feedback_template(
		["highest_payoff_for_choice(10, 'B').", "highest_payoff(10)", "lower(10, 35)",
		 "lowest_min_payoff_choice('B')", "highest_min_payoff_choice('B')", "highest_possible_payoff(10)",
		 "lowest_possible_payoff(10)", "highest_joint_payoff('B', 'R')", "lowest_joint_payoff('B', 'R')"])
	logger.debug(formatted)


if __name__ == '__main__':
	test("DATA/TEMPLATES/", "DATA/PROMPTS")
