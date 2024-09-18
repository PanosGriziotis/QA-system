import json
import pandas as pd
import matplotlib.pyplot as plt

# Function to load the list of dictionaries from a file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to process the data and calculate average relevance score for each broad intent
def process_data(data, pipeline_type):
    processed_data = []
    
    for entry in data:
        intent = entry['intent']

        if intent in ["out_of_scope/is_quarantine_effective", "out_of_scope/covid_and_weather", "out_of_scope/cleaning", "out_of_scope/mortality_rate"]:
            broad_intent = "FAQ"
            
        else:
            sub_intent = intent.split('/')[0]
            if sub_intent == "out_of_scope":
                broad_intent= "Out-of-Scope"
            elif sub_intent in ['EODY_faq', 'vaccines' ,'mask_faq']:
                broad_intent = "FAQ"

        relevance_score = entry['answers'][0]['meta']['answer_relevance_ragas']
        #avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        processed_data.append({'intent_category': broad_intent, 
                               'avg_relevance_score': relevance_score, 
                               'type': pipeline_type})
    
    return processed_data

# Function to create side-by-side box plots for comparison
def create_comparative_box_plot(data1, data2):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Combine both datasets and group them
    combined_df = pd.concat([df1, df2])
    
    print (combined_df)
    # List of unique broad intents
    intent_categories = combined_df['intent_category'].unique()
    print (intent_categories)
    
    # Plot side-by-side box plots for each broad intent
    plt.figure(figsize=(12, 6))
    
    for i, intent in enumerate(intent_categories):
        plt.subplot(1, len(intent_categories), i+1)
        
        # Extract data for the specific intent
        intent_data = combined_df[combined_df['intent_category'] == intent]
        
        # Group data by bot for that specific intent
        bot_groups = [intent_data[intent_data['type'] == bot]['avg_relevance_score'] 
                      for bot in ['Extractive', 'Generative']]
        
        box_colors = ['skyblue', 'orange']
        box = plt.boxplot(bot_groups, patch_artist=True, labels=['Extractive', 'Generative'])

        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)

        plt.title(intent)
        plt.ylim(0, 1)  # Assuming relevance score ranges between 0 and 1
    
        if i == 0:
            plt.ylabel('Answer Relevance')
    # General plot settings
    plt.savefig("./box_plots.png")
    plt.close()

# Example usage
file1_path = 'extractive_10/final_results_threashold/extractive_10_10_full.json'  # Replace with the actual file path for TruthBot
file2_path = 'rag_15/final_results_threashold/rag_15_100_full.json'  # Replace with the actual file path for IFCNBot

# Load the data from both files
data1 = load_data(file1_path)
data2 = load_data(file2_path)

# Process the data to extract broad intent categories and average relevance scores
processed_data1 = process_data(data1, "Extractive")
processed_data2 = process_data(data2, "Generative")

# Create the comparative box plot
create_comparative_box_plot(processed_data1, processed_data2)