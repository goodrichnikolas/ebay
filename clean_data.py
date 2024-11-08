import os
import ollama
import pandas as pd
import logging
from typing import Optional, Dict
from datetime import datetime

class InformationExtractor:
    def __init__(self, host='http://172.20.128.1:11434', model='llama3.1', debug=False):
        self.model = model
        os.environ['OLLAMA_HOST'] = host
        self.debug = debug
        
        # Set up logging
        log_filename = f'information_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing InformationExtractor with model: {model}, host: {host}")
        
    def create_prompt(self, text: str, instruction: str) -> str:
        """
        Create a prompt based on the instruction and input text
        
        Args:
            text: The input text to process
            instruction: The instruction for what to extract/how to process the text
        """
        self.logger.debug(f"Creating prompt for text: {text}")
        prompt = f"""{instruction}
        
        Text to process: {text}
        """
        self.logger.debug(f"Generated prompt: {prompt}")
        return prompt
    
    def process_text(self, text: str, instruction: str) -> str:
        """Process a single text input"""
        self.logger.debug(f"Processing text: {text}")
        try:
            prompt = self.create_prompt(text, instruction)
            self.logger.debug(f"Sending request to Ollama")
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            result = response['message']['content'].strip()
            self.logger.debug(f"Received response: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error processing text: {e}", exc_info=True)
            return "Error"
    
    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         input_column: str,
                         output_column: str,
                         instruction: str,
                         row_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Process a dataframe and add results to a new column
        
        Args:
            df: Input DataFrame
            input_column: Column containing text to process
            output_column: Column to store results
            instruction: Instruction for what to extract/how to process the text
            row_limit: Optional limit on number of rows to process (None for all rows)
        """
        if input_column not in df.columns:
            error_msg = f"Input column '{input_column}' not found in DataFrame"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Starting DataFrame processing with {len(df)} rows")
        if row_limit:
            self.logger.info(f"Row limit active: processing only {row_limit} rows")
            df_to_process = df.head(row_limit)
        else:
            df_to_process = df
            
        # Create the output column if it doesn't exist
        total_rows = len(df_to_process)
        df[output_column] = None  # Initialize column
        
        for idx, row in df_to_process.iterrows():
            self.logger.info(f"Processing row {idx + 1}/{total_rows}")
            df.at[idx, output_column] = self.process_text(row[input_column], instruction)
            
        self.logger.info("DataFrame processing completed")
        return df

def normalize_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all string/object columns in the DataFrame to lowercase
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with all string values converted to lowercase
    """
    # Create a copy to avoid modifying the original DataFrame
    df_normalized = df.copy()
    
    # Get all string/object columns
    string_columns = df_normalized.select_dtypes(include=['object']).columns
    
    # Convert each string column to lowercase
    for column in string_columns:
        df_normalized[column] = df_normalized[column].str.lower()
    
    return df_normalized

# Example usage
if __name__ == "__main__":
    # Configure logging for the main script
    logging.info("Starting main script execution")
    
    try:
        # Load data
        df = pd.read_csv('./output/dbz score graded_20241108.csv')
        logging.info(f"Loaded DataFrame with {len(df)} rows")
        
        assert df is not None, "Error loading DataFrame"
        
        # Example extraction parameters
        extraction_params = {
            'input_column': 'title',  # Column containing the text to process
            'output_column': 'character',  # Column to store the extracted information
            'instruction': """Extract only the Dragon Ball Z character name from this text.
                            If no character is mentioned, return 'None'.
                            Only return the character name, nothing else.""",
            'row_limit': None  # Set to a number to limit rows processed, None for all rows
        }
        
        # Initialize extractor with debug mode
        extractor = InformationExtractor(debug=True)
        
        # Process the DataFrame
        df_processed = extractor.process_dataframe(
            df=df,
            **extraction_params
        )
        
        # Normalize case for all columns
        df_processed = normalize_case(df_processed)
        
        logging.info("\nProcessed DataFrame:")
        print(df_processed[['title', 'character']])
        
        # Optionally save the processed DataFrame
        output_file = f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_processed.to_csv(output_file, index=False)
        logging.info(f"Saved processed data to {output_file}")
        
    except Exception as e:
        logging.error(f"An error occurred in main script: {e}", exc_info=True)