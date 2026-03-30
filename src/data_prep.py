import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    print("1. Fetching dataset from Hugging Face...")
    # Loading the Bitext customer support dataset
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    
    # Convert 'train' split into a Pandas DataFrame
    df = dataset['train'].to_pandas()
    
    # Keep only the columns we need (FIXED: 'instruction' instead of 'utterance')
    df = df[['instruction', 'intent']]
    df.rename(columns={'instruction': 'text'}, inplace=True)
    
    # Ensure directories exist
    os.makedirs(os.path.join('..', 'data', 'raw'), exist_ok=True)
    os.makedirs(os.path.join('..', 'data', 'processed'), exist_ok=True)
    
    # Save raw data
    raw_path = os.path.join('..', 'data', 'raw', 'full_dataset.csv')
    df.to_csv(raw_path, index=False)

    print("2. Filtering down to 10 distinct intents...")
    target_intents = [
        'check_refund_policy', 
        'report_fraud', 
        'cancel_order', 
        'track_refund', 
        'change_account_password', 
        'change_shipping_address', 
        'contact_customer_service', 
        'check_cancellation_fee',   
        'payment_issue',            
        'delivery_options'          
    ]
    
    filtered_df = df[df['intent'].isin(target_intents)].copy()
    print(f"Data remaining after filtering: {len(filtered_df)} rows.")

    print("3. Splitting the data (80% Train, 10% Val, 10% Test)...")
    # 80% Train, 20% Temp
    train_df, temp_df = train_test_split(
        filtered_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=filtered_df['intent']
    )
    
    # Split the 20% Temp into 50% Val / 50% Test (10% of total each)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_df['intent']
    )

    print("4. Saving splits to data\\processed\\...")
    processed_dir = os.path.join('..', 'data', 'processed')
    
    train_df.to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test.csv'), index=False)
    
    print("\n--- Summary ---")
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print("Data engineering complete!")

if __name__ == "__main__":
    # Ensure the script is run from the src directory
    if not os.path.exists(os.path.join('..', 'data')):
        print("Please run this script from inside the 'src' folder.")
    else:
        prepare_data()