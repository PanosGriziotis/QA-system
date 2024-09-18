import random
import collections

# Initialize the dictionary with default empty lists for each sub-intent
queries_faq = collections.defaultdict(list)

file_names = ['EODY_faq.txt', 'vaccines.txt', 'mask_faq.txt']

for file_name in file_names:
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                query, sub_intent = line.split('\t')
                queries_faq[sub_intent].append(query)


with open('out_of_scope_general_filtered.txt', 'r') as fp:
    queries_out_of_scope = [line.strip().split('\t')[0] for line in fp.readlines()]

queries_faq = dict(queries_faq)

# Set the number of queries
num_out_of_scope = 82
queries_per_sub_intent = 1

# Select queries for "out_of_scope"
selected_out_of_scope = random.sample(queries_out_of_scope, num_out_of_scope)

# Create a list of tuples for "out_of_scope" queries with label
selected_out_of_scope = [(query, 'out_of_scope') for query in selected_out_of_scope]

# Select queries for "faq"
selected_faq = []
for sub_intent in queries_faq:
    selected_queries = random.sample(queries_faq[sub_intent], queries_per_sub_intent)
    selected_faq.extend((query, sub_intent) for query in selected_queries)

# Combine and shuffle the final dataset
final_dataset = selected_out_of_scope + selected_faq
random.shuffle(final_dataset)

# Save the final dataset to a text file
with open('final_dataset.txt', 'w') as outfile:
    for query, intent in final_dataset:
        outfile.write(f"{query}\t{intent}\n")