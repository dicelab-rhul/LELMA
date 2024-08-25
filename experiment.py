import random
import configparser
import os
import logging
from src.setup_logger import logger
from src.prompt_maker import PromptMaker
from llms.gpt4 import GPT4
from llms.gemini import Gemini
from src.solver import Solver
from src.lelma import LELMA


def main():
	logging.debug('Test')

	# Read experiment parameters
	config = configparser.ConfigParser()
	config.read("DATA/CONFIG/full_experiment.ini")

	GAME_DIR = config.get("Paths", "GAME_DIR")
	PROMPT_DIR = config.get("Paths", "PROMPT_DIR")
	OUT_DIR = config.get("Paths", "OUT_DIR")
	if not os.path.exists(OUT_DIR):
		os.makedirs(OUT_DIR)

	solver_path = config.get("Paths", "SOLVER_PATH")
	repetitions = config.getint("General", "repetitions")
	games = config.get("Games", "types").split(";")
	max_attempts = config.getint("General", "max_attempts")

	# Instantiate components
	translation_prompt_fname = config.get("Prompts", "translation")
	feedback_prompt_fname = config.get("Prompts", "feedback")
	prompt_maker = PromptMaker(translation_template=translation_prompt_fname, feedback_template=feedback_prompt_fname)
	solver = Solver(solver_path)

	gpt4_reasoner = GPT4(save_history=True, model="gpt-4o-2024-05-13")
	gemini_reasoner = Gemini(save_history=True)
	llm_translator = GPT4()

	reasoners = [gpt4_reasoner, gemini_reasoner]

	# Read instruction prompt from files
	instruction_prompts_fnames = config.get("Prompts", "instruction").split(";")

	# Main experimental loop
	# For each LLM-reasoner
	for reasoner in reasoners:
		lelma = LELMA(reasoner, llm_translator, solver, prompt_maker, max_attempts)
		# Repeat repetitions times
		for rep in range(repetitions):
			# For each prompt phrasing version
			for prompt_file in instruction_prompts_fnames:
				# Read prompt from file
				prompt_template = prompt_maker.read_prompt_from_file(os.path.join(PROMPT_DIR, prompt_file))
				# For each game type
				for game_type in games:
					# For each payoff matrix
					mat_path = os.path.join(GAME_DIR, game_type)
					for mat_file in config.get("Matrices", game_type).split(";"):
						payoff_matrix_file = os.path.join(mat_path, mat_file)
						prompt = prompt_maker.fill_instruction_template(prompt_template, payoff_matrix_file)
						payoff_matrix = prompt_maker.convert_pl_to_dict(payoff_matrix_file)

						fout_name = os.path.join(OUT_DIR,
											 f"{reasoner.get_name()}_{game_type}_{prompt_file[:-4]}_{payoff_matrix['UL'][0]}_{payoff_matrix['UR'][0]}_{payoff_matrix['DL'][0]}_{payoff_matrix['DR'][0]}_{rep}_{random.randint(1000, 9999)}.txt")

						try:  # We don't want to break the loop
							lelma.reason(payoff_matrix_file, prompt, fout_name)

						except Exception as error:
							logger.exception(error)


if __name__ == '__main__':
	main()
