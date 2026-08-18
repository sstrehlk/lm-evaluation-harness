"""
Utilities for MMLU tasks
"""
from functools import partial
import datasets
import random


# All MMLU subjects/subcategories
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions"
]


def shuffle_docs(dataset, seed=0):
    """
    Shuffle the order of documents in the dataset with a fixed seed.
    
    Args:
        dataset: Dataset or list of documents
        seed: Random seed for reproducibility (default: 0)
        
    Returns:
        Shuffled Dataset with same structure as input
    """
    docs = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(docs)
    # Return as Dataset to preserve features metadata
    return datasets.Dataset.from_list(docs)


# Partial function with seed=117 for mmlu_combined_shuffled task
shuffle_docs_seed117 = partial(shuffle_docs, seed=117)


def load_all_mmlu_combined(**kwargs):
    """
    Load and combine all MMLU subjects into a single dataset.
    This allows using a global limit across all 57 subcategories.
    
    Returns:
        Dictionary with 'test' and 'dev' splits containing all subjects combined
    """
    all_test_docs = []
    all_dev_docs = []
    
    print(f"Loading all {len(MMLU_SUBJECTS)} MMLU subjects...")
    
    # Load all subjects and combine them
    for idx, subject in enumerate(MMLU_SUBJECTS):
        try:
            # Load test split
            test_data = datasets.load_dataset(
                "cais/mmlu",
                name=subject,
                split='test'
            )
            
            # Load dev split for fewshot examples
            dev_data = datasets.load_dataset(
                "cais/mmlu",
                name=subject,
                split='dev'
            )
            
            # Add subject name to each document for tracking
            for doc in test_data:
                doc_dict = dict(doc)
                doc_dict['subject'] = subject
                all_test_docs.append(doc_dict)
            
            for doc in dev_data:
                doc_dict = dict(doc)
                doc_dict['subject'] = subject
                all_dev_docs.append(doc_dict)
            
            if (idx + 1) % 10 == 0:
                print(f"  Loaded {idx + 1}/{len(MMLU_SUBJECTS)} subjects...")
                
        except Exception as e:
            print(f"Warning: Could not load subject {subject}: {e}")
            continue
    
    print(f"Total documents - Test: {len(all_test_docs)}, Dev: {len(all_dev_docs)}")
    
    # Create Dataset objects
    return {
        'test': datasets.Dataset.from_list(all_test_docs),
        'dev': datasets.Dataset.from_list(all_dev_docs)
    }

