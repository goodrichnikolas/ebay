import os
import ollama
import pandas as pd
import logging
from typing import Optional, Dict, List
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
    
    def process_dataframe_multiple(self, 
                                 df: pd.DataFrame, 
                                 extraction_params_list: List[Dict],
                                 row_limit: Optional[int] = None) -> pd.DataFrame:
        """
        Process a dataframe with multiple extraction parameters
        
        Args:
            df: Input DataFrame
            extraction_params_list: List of dictionaries containing extraction parameters
                Each dict should have:
                - input_column: Column containing text to process
                - output_column: Column to store results
                - instruction: Instruction for what to extract
            row_limit: Optional limit on number of rows to process (None for all rows)
        """
        df_result = df.copy()
        
        if row_limit:
            self.logger.info(f"Row limit active: processing only {row_limit} rows")
            df_to_process = df_result.head(row_limit)
        else:
            df_to_process = df_result
            
        total_rows = len(df_to_process)
        
        for params in extraction_params_list:
            input_column = params['input_column']
            output_column = params['output_column']
            instruction = params['instruction']
            
            if input_column not in df_result.columns:
                error_msg = f"Input column '{input_column}' not found in DataFrame"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"Processing extraction for output column: {output_column}")
            df_result[output_column] = None  # Initialize column
            
            for idx, row in df_to_process.iterrows():
                self.logger.info(f"Processing row {idx + 1}/{total_rows} for {output_column}")
                df_result.at[idx, output_column] = self.process_text(row[input_column], instruction)
        
        self.logger.info("DataFrame processing completed")
        return df_result

def normalize_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all string/object columns in the DataFrame to lowercase
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with all string values converted to lowercase
    """
    df_normalized = df.copy()
    string_columns = df_normalized.select_dtypes(include=['object']).columns
    
    for column in string_columns:
        df_normalized[column] = df_normalized[column].str.lower()
    
    return df_normalized

def delete_row_on_length(df: pd.DataFrame, column: str, length: int) -> pd.DataFrame:
    """
    Delete rows from the DataFrame where the length of the specified column is greater than a certain value
    
    Args:
        df: Input DataFrame
        column: Column to check for length
        length: Minimum length for the column value
    
    Returns:
        DataFrame with rows removed where the column value is greater than the specified length
    """
    df_filtered = df[df[column].str.len() <= length]
    return df_filtered

# Example usage
if __name__ == "__main__":
    logging.info("Starting main script execution")
    
    try:
        # Load data
        df = pd.read_csv('./output/dbz score graded_20241108.csv')
        logging.info(f"Loaded DataFrame with {len(df)} rows")
        
        assert df is not None, "Error loading DataFrame"
        
        # Define multiple extraction parameters
        extraction_params_list = [
            {
                'input_column': 'title',
                'output_column': 'character',
                'instruction': """Extract only the Dragon Ball Z character name from this text.
                                If no character is mentioned, return 'None'.
                                Only return the character name, nothing else."""
            },
            {
                'input_column': 'title',
                'output_column': 'psa_grade',
                'instruction': """Extract only the PSA grade number from this text.
                                Return only the number (e.g., '9.8', '7.0').
                                If no PSA grade is mentioned, return 'None'.
                                Do not include 'PSA' or any other text."""
            }
        ]
        
        # Initialize extractor with debug mode
        extractor = InformationExtractor(debug=True)
        
        # Process the DataFrame with multiple parameters
        df_processed = extractor.process_dataframe_multiple(
            df=df,
            extraction_params_list=extraction_params_list
        )
        
        # Normalize case for all columns
        df_processed = normalize_case(df_processed)
        
        # Split character column if needed
        df_processed[['character', 'character2']] = df_processed['character'].str.split(r'\n', n=1, expand=True)
        
        logging.info("\nProcessed DataFrame:")
        print(df_processed[['title', 'character', 'psa_grade']])
        
        #For columns character, character2, delete any rows longer than 25 characters
        df_processed = delete_row_on_length(df_processed, 'character', 25)
        
        # Save the processed DataFrame
        output_file = f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_processed.to_csv(output_file, index=False)
        logging.info(f"Saved processed data to {output_file}")
        
    except Exception as e:
        logging.error(f"An error occurred in main script: {e}", exc_info=True)