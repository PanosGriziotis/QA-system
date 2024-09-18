import pandas as pd
import os
# Load the CSV file containing the summary scores
 # Replace with your actual file path

def rank_expirements (summary_file_path):
    summary_df = pd.read_csv(summary_file_path)

    # Handle missing values in the columns by filling with a neutral value (like 0)
    summary_df['avg_answer_relevance'].fillna(0, inplace=True)
    summary_df['unanswerable_percentage'].fillna(100, inplace=True)  # High percentage if unanswered questions are missing

    # Sort experiments by avg_answer_relevance (descending) and unanswerable_percentage (ascending)
    sorted_df = summary_df.sort_values(
        by=['avg_answer_relevance', 'unanswerable_percentage'], 
        ascending=[False, True]
    )

    # Add a rank column based on sorting
    sorted_df['rank'] = range(1, len(sorted_df) + 1)

    output_file_path = os.path.join(os.path.dirname(summary_file_path), 'ranked_experiments_answer_relevance.csv')

    sorted_df.to_csv(output_file_path, index=False)

    print(f"Ranked experiments saved to {output_file_path}")



# Save the ranked experiments to a new CSV file
summary_file_path = "final_results_rerun/rag_processed/final_results_threashold/summary_scores_per_experiment.csv"
rank_expirements(summary_file_path=summary_file_path)

