Columns contain the following information:

file: log file name
game: can be "hd" (Hawk-Dove), "pd" (Prisoner's Dilemma), or "sh" (Stag Hunt)
attempts: number of reasoning attempts, ranges from 1 to 5
orig_choice: action ("B" or "R") selected in the first attempt
final_choice: action ("B" or "R") selected in the last attempt
orig_cor: correctness of the initial reasoning attempt (0 or 1)
fin_cor: correctness of the final reasoning attempt (0 or 1)
con_mat: confusion matrix entry: "TP" (True Positive), "FP" (False Positive), "TN" (True Negative), "FN" (False Negative)
filed_queries: list of the failed queries
orig_cor_pred: correctness of the initial reasoning attempt (0 or 1), based on the failed queries
fin_cor_pred: correctness of the final reasoning attempt (0 or 1), based on the failed queries
instr_tokens: number of tokens in the instructions
gen_tokens: number of generated tokens
time: time in seconds
