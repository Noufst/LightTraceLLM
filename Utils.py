import datetime
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split

def get_random_samples(
    df: pd.DataFrame,
    num_samples: int = 2,
    random_state: Optional[int] = None
) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    """Extract random balanced samples for few-shot learning.

    Args:
        df: Training dataframe with 'label' column
        num_samples: Number of samples to extract (split evenly between labels)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (examples_list, selected_examples_dataframe)
        where examples is list of (prompt, response) tuples
    """

    num_samples_per_label = num_samples // 2

    examples = []
    selected_examples = []
 
    # Extract random samples for label 0
    random_rows_label_0 = df[df['label'] == 0].sample(n=num_samples_per_label, random_state=random_state)
    for idx, row in random_rows_label_0.iterrows():
        example = ("(1) " + row["source_content"] + "\n\n" + "(2) " + row["target_content"] + "\n\n")
        examples.append((example, "No"))
        selected_examples.append({
            "source_content": row["source_content"],
            "target_content": row["target_content"],
            "label": "No"
        })

    # Extract random samples for label 1
    random_rows_label_1 = df[df['label'] == 1].sample(n=num_samples_per_label, random_state=random_state)
    for idx, row in random_rows_label_1.iterrows():
        example = ("(1) " + row["source_content"] + "\n\n" + "(2) " + row["target_content"] + "\n\n")
        examples.append((example, "Yes"))
        selected_examples.append({
            "source_content": row["source_content"],
            "target_content": row["target_content"],
            "label": "Yes"
        })

    # Combine selected examples into DataFrame
    selected_examples_df = pd.concat([random_rows_label_0, random_rows_label_1], ignore_index=True)

    return examples, selected_examples_df

def get_random_samples_imbalanced(df, num_samples=1, random_state=None):
    """
    Extract random samples from a DataFrame for few-shot learning.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data.
    - num_samples (int): The number of samples to extract (default is 1).
    - random_state (int, optional): Random seed for reproducibility.

    Returns:
    - examples (list of tuples): List of tuples where each tuple contains an example string and its corresponding response.
    - df (pd.DataFrame): The DataFrame with the extracted rows removed.
    """

    examples = []
    selected_examples = []

    print(df.columns)
    # Extract random samples for label 0
    random_rows = df.sample(n=num_samples, random_state=random_state)
    for idx, row in random_rows.iterrows():
        example = ("(1) " + row["source_content"] + "\n\n" + "(2) " + row["target_content"] + "\n\n")
        label = "No" if row["label"] == 0 else "Yes"
        print("*********")
        print(label)
        examples.append((example, label))
        selected_examples.append({
            "source_content": row["source_content"],
            "target_content": row["target_content"],
            "label": row["label"]
        })

    return examples, random_rows

def get_diverse_samples(
    train_df: pd.DataFrame,
    strategy: str = 'sum',
    num_samples: int = 2
) -> Tuple[List[Tuple[str, str]], pd.DataFrame, Optional[List[int]]]:
    """Extract diverse samples using embedding-based selection.

    Selects examples that maximize diversity in the embedding space,
    providing better coverage for few-shot learning than random sampling.

    Args:
        train_df: Training data with embeddings
        strategy: Selection strategy ('sum', 'max', or 'avg')
        num_samples: Number of samples to select

    Returns:
        Tuple of (examples, remaining_df, selected_indices)
    """
    print("Starting retrieving diverse samples using global scoring approach...")

    num_samples_per_label = num_samples // 2
    examples = []
    train_df_with_similarity_scores = train_df.copy()

    # Parse embeddings
    train_df_with_similarity_scores['combined_content_embeddings'] = train_df_with_similarity_scores['combined_content_embeddings'].apply(json.loads)
    embeddings = np.vstack(train_df_with_similarity_scores['combined_content_embeddings'].values)

    print("Computing pairwise similarity...")
    similarity_matrix = cosine_similarity(embeddings)

    # Separate indices by label
    positive_indices = train_df_with_similarity_scores[train_df_with_similarity_scores['label'] == 1].index.tolist()
    negative_indices = train_df_with_similarity_scores[train_df_with_similarity_scores['label'] == 0].index.tolist()

    if num_samples_per_label > len(positive_indices) or num_samples_per_label > len(negative_indices):
        raise ValueError("Requested number of samples exceeds available data for one or both labels.")

    # Find the pair of indices with the lowest similarity score where labels are different
    best_pair = None
    best_score = float('inf')
    for pos_idx in positive_indices:
        for neg_idx in negative_indices:
            similarity = similarity_matrix[pos_idx, neg_idx]
            if similarity < best_score:
                best_score = similarity
                best_pair = (pos_idx, neg_idx)

    if not best_pair:
        raise ValueError("No dissimilar pairs with different labels found.")

    print(f"Most dissimilar pair: {best_pair} with similarity score {best_score}")
    selected_indices = list(best_pair)
    positive_selected = sum(train_df_with_similarity_scores.loc[idx, 'label'] == 1 for idx in best_pair)
    negative_selected = sum(train_df_with_similarity_scores.loc[idx, 'label'] == 0 for idx in best_pair)
    # Combine both positive and negative pools
    remaining_indices = [idx for idx in train_df_with_similarity_scores.index if idx not in selected_indices]

    while positive_selected < num_samples_per_label or negative_selected < num_samples_per_label:
        best_candidate = None
        best_score = float('inf')
        best_label = None

        for idx in remaining_indices:
            # Ensure balance: skip if selecting this example would unbalance the output
            if train_df_with_similarity_scores.loc[idx, 'label'] == 1 and positive_selected >= num_samples_per_label:
                continue
            if train_df_with_similarity_scores.loc[idx, 'label'] == 0 and negative_selected >= num_samples_per_label:
                continue

            # Compute similarity with already selected examples
            if strategy == "sum":
                candidate_score = sum(
                    similarity_matrix[idx, selected_idx] for selected_idx in selected_indices
                ) #if selected_indices else 0
            elif strategy == "max":
                candidate_score = max(
                    similarity_matrix[idx, selected_idx] for selected_idx in selected_indices
                ) #if selected_indices else 0

            # Track the best candidate
            if candidate_score < best_score:
                best_candidate = idx
                best_score = candidate_score
                best_label = train_df_with_similarity_scores.loc[idx, 'label']

        # Add the best candidate to the selected indices
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
        if best_label == 1:
            positive_selected += 1
        elif best_label == 0:
            negative_selected += 1

    print("Global scoring selection complete.")
    
    # Save the selected examples
    selected_df = train_df_with_similarity_scores.loc[selected_indices]
    print(selected_df)
    ############## order #############
    # selected_df = selected_df.sort_values(by='label', ascending=False)
    # print("After Sorting..............")
    # print(selected_df)
    ##################################

    # Compute similarity scores for the selected subset
    print("Generating similarity scores for selected examples...")
    selected_pairs = [(i, j) for i in selected_indices for j in selected_indices if i < j]
    similarity_scores_df = pd.DataFrame(
        [(train_df_with_similarity_scores.iloc[i]['source_content'], train_df_with_similarity_scores.iloc[i]['target_content'], train_df_with_similarity_scores.iloc[i]['label'],
          train_df_with_similarity_scores.iloc[j]['source_content'], train_df_with_similarity_scores.iloc[j]['target_content'], train_df_with_similarity_scores.iloc[j]['label'],
          similarity_matrix[i, j])
         for i, j in selected_pairs],
        columns=["source_content_1", "target_content_1", "label_1",
                 "source_content_2", "target_content_2", "label_2",
                 "similarity"]
    )

    # Map labels to "No" for 0 and "Yes" for 1
    selected_df["label"] = selected_df["label"].replace({0: "No", 1: "Yes"})
    # Add the selected pairs to the list of examples  
    for _, row in selected_df.iterrows():
        example = ("(1) " + row['source_content'] + "\n\n" + "(2) " + row['target_content'] + "\n\n")
        examples.append((example, row['label']))

    print(len(examples))
    for x in examples:
        print(x[1])
    return examples, selected_df, similarity_scores_df

def get_diverse_samples_imbalanced(train_df, strategy='sum', num_samples=1):
    print("Starting retrieving diverse samples using global scoring approach...")

    examples = []

    train_df_with_similarity_scores = train_df.copy()

    # Parse embeddings
    train_df_with_similarity_scores['combined_content_embeddings'] = train_df_with_similarity_scores['combined_content_embeddings'].apply(json.loads)
    embeddings = np.vstack(train_df_with_similarity_scores['combined_content_embeddings'].values)

    print("Computing pairwise similarity...")
    similarity_matrix = cosine_similarity(embeddings)

    if num_samples > len(train_df_with_similarity_scores.index.tolist()):
        raise ValueError("Requested number of samples exceeds available data.")

    print("Identifying the most dissimilar pair...")
    # Find the most dissimilar pair
    i, j = np.unravel_index(np.argmin(similarity_matrix + np.eye(len(similarity_matrix)) * 2), similarity_matrix.shape)
    selected_indices = [i, j]
    remaining_indices = [idx for idx in train_df_with_similarity_scores.index if idx not in selected_indices]

    print(f"Most dissimilar pair: {i}, {j}")
    print("Starting global scoring selection...")

    num_selected_indices = 2
    while num_selected_indices < num_samples:
        best_candidate = None
        best_score = float('inf')

        for idx in remaining_indices:

            # Compute similarity with already selected examples
            if strategy == "sum":
                candidate_score = sum(
                    similarity_matrix[idx, selected_idx] for selected_idx in selected_indices
                ) if selected_indices else 0
            elif strategy == "max":
                candidate_score = max(
                    similarity_matrix[idx, selected_idx] for selected_idx in selected_indices
                ) if selected_indices else 0

            # Track the best candidate
            if candidate_score < best_score:
                best_candidate = idx
                best_score = candidate_score

        # Add the best candidate to the selected indices
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
        num_selected_indices += 1

    print("Global scoring selection complete.")
    
    # Save the selected examples
    selected_df = train_df_with_similarity_scores.loc[selected_indices]

    # Compute similarity scores for the selected subset
    print("Generating similarity scores for selected examples...")
    selected_pairs = [(i, j) for i in selected_indices for j in selected_indices if i < j]
    similarity_scores_df = pd.DataFrame(
        [(train_df_with_similarity_scores.iloc[i]['source_content'], train_df_with_similarity_scores.iloc[i]['target_content'], train_df_with_similarity_scores.iloc[i]['label'],
          train_df_with_similarity_scores.iloc[j]['source_content'], train_df_with_similarity_scores.iloc[j]['target_content'], train_df_with_similarity_scores.iloc[j]['label'],
          similarity_matrix[i, j])
         for i, j in selected_pairs],
        columns=["source_content_1", "target_content_1", "label_1",
                 "source_content_2", "target_content_2", "label_2",
                 "similarity"]
    )

    # Map labels to "No" for 0 and "Yes" for 1
    train_df_with_similarity_scores["label"] = train_df_with_similarity_scores["label"].replace({0: "No", 1: "Yes"})
    # Add the selected pairs to the list of examples  
    for i, j in selected_pairs:
        example = ("(1) " + train_df_with_similarity_scores.iloc[i]['source_content'] + "\n\n" + "(2) " + train_df_with_similarity_scores.iloc[i]['target_content'] + "\n\n")
        examples.append((example, train_df_with_similarity_scores.iloc[i]['label']))
        example = ("(1) " + train_df_with_similarity_scores.iloc[j]['source_content'] + "\n\n" + "(2) " + train_df_with_similarity_scores.iloc[j]['target_content'] + "\n\n")
        examples.append((example, train_df_with_similarity_scores.iloc[j]['label']))
    
    return examples, selected_df, similarity_scores_df

def log_conversation(user_input, response, log_file):
    
    """Logs the conversation to a JSON file."""

     # Prepare the log entry in Markdown format
    timestamp = datetime.datetime.now().isoformat()
    formatted_log_entry = f"""
### Timestamp: {timestamp}
**User Input:**
{user_input.replace('`', '')}

**Response:**
{response.replace('`', '')}

---
"""

    # Write the formatted log entry to the Markdown file
    with open(log_file, 'a') as file:
        file.write(formatted_log_entry)



