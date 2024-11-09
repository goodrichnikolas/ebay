import os
import ollama
import pandas as pd
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import re
from functools import wraps
import time

class ModelType(Enum):
    """Supported LLM models"""
    LLAMA = "llama3.1"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"

@dataclass
class ExtractionParams:
    """Configuration for text extraction tasks"""
    input_column: str
    output_column: str
    instruction: str

class ExtractionError(Exception):
    """Custom exception for extraction-related errors"""
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = delay * (2 ** attempt)
                    logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        return wrapper
    return decorator

class InformationExtractor:
    """Extract structured information from text using LLM models"""
    
    def __init__(
        self,
        host: str = 'http://172.20.128.1:11434',
        model: ModelType = ModelType.LLAMA,
        debug: bool = False,
        log_dir: Optional[Path] = None
    ):
        """
        Initialize the Information Extractor.
        
        Args:
            host: Ollama API host address
            model: ModelType enum specifying which model to use
            debug: Enable debug logging
            log_dir: Directory for log files
        """
        self.model = model.value
        os.environ['OLLAMA_HOST'] = host
        self.debug = debug
        
        # Set up logging
        log_dir = log_dir or Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_filename = log_dir / f'information_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing InformationExtractor with model: {model.value}, host: {host}")

    def create_prompt(self, text: str, instruction: str) -> str:
        """Create a formatted prompt combining instruction and input text"""
        self.logger.debug(f"Creating prompt for text: {text[:100]}...")
        return f"{instruction.strip()}\n\nText to process: {text.strip()}"

    @retry_on_failure(max_retries=3)
    def process_text(self, text: str, instruction: str) -> str:
        """
        Process a single text input with retry mechanism.
        
        Raises:
            ExtractionError: If text processing fails after all retries
        """
        if not text or not isinstance(text, str):
            return "None"
            
        self.logger.debug(f"Processing text: {text[:100]}...")
        try:
            prompt = self.create_prompt(text, instruction)
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            raise ExtractionError(f"Failed to process text: {str(e)}")

    def process_dataframe_multiple(
        self,
        df: pd.DataFrame,
        extraction_params_list: List[ExtractionParams],
        row_limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process multiple extraction tasks on a DataFrame.
        
        Args:
            df: Input DataFrame
            extraction_params_list: List of ExtractionParams objects
            row_limit: Optional limit on number of rows to process
            
        Returns:
            DataFrame with extracted information in new columns
        
        Raises:
            ValueError: If input columns don't exist in DataFrame
        """
        df_result = df.copy()
        df_to_process = df_result.head(row_limit) if row_limit else df_result
        total_rows = len(df_to_process)
        
        for params in extraction_params_list:
            if params.input_column not in df_result.columns:
                raise ValueError(f"Input column '{params.input_column}' not found in DataFrame")
            
            self.logger.info(f"Processing extraction for output column: {params.output_column}")
            df_result[params.output_column] = pd.NA
            
            for idx, row in df_to_process.iterrows():
                self.logger.info(f"Processing row {idx + 1}/{total_rows} for {params.output_column}")
                try:
                    result = self.process_text(row[params.input_column], params.instruction)
                    df_result.at[idx, params.output_column] = result
                except ExtractionError as e:
                    self.logger.error(f"Error processing row {idx}: {str(e)}")
                    df_result.at[idx, params.output_column] = "Error"
        
        return df_result

class DataFrameProcessor:
    """Utility class for DataFrame operations"""
    
    @staticmethod
    def normalize_case(df: pd.DataFrame) -> pd.DataFrame:
        """Convert all string columns to lowercase"""
        df_normalized = df.copy()
        string_columns = df_normalized.select_dtypes(include=['object']).columns
        
        for column in string_columns:
            df_normalized[column] = df_normalized[column].str.lower()
        
        return df_normalized
    
    @staticmethod
    def delete_row_on_length(df: pd.DataFrame, column: str, max_length: int) -> pd.DataFrame:
        """Remove rows where column value exceeds maximum length"""
        return df[df[column].str.len() <= max_length]
    
    @staticmethod
    def extract_price_shipping(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Extract numerical price from text"""
        df_price = df.copy()
        df_price[column] = df_price[column].astype(str)  # Ensure the column is of string type
        df_price['shipping'] = pd.to_numeric(
            df_price[column].str.extract(r'\$(\d+\.\d+)')[0],
            errors='coerce'
        )
        # Replace empty rows with 0
        df_price['shipping'] = df_price['shipping'].fillna(0)
        return df_price
    
    @staticmethod
    def extract_sold_date(df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Extract and standardize sale date"""
        df_date = df.copy()
        date_pattern = r'sold\s+(\w{3}\s\d{1,2},\s\d{4})'
        df_date['date'] = pd.to_datetime(
            df_date[column].str.extract(date_pattern)[0],
            format='%b %d, %Y',
            errors='coerce'
        )
        return df_date

def main():
    """Main execution function"""
    logging.info("Starting main script execution")
    
    try:
        # Load data by selecting the most recent csv from ./output/
        output_dir = Path('output')
        output_files = list(output_dir.glob('*.csv'))
        input_file = max(output_files, key=os.path.getctime)
        df = pd.read_csv(input_file)
        logging.info(f"Loaded DataFrame with {len(df)} rows from {input_file}")
        
        # Define extraction parameters
        extraction_params = [
            ExtractionParams(
                input_column='title',
                output_column='character',
                instruction="""Extract only the Dragon Ball Z character name from this text.
                             If no character is mentioned, return 'None'.
                             Only return the character name, nothing else."""
            ),
            ExtractionParams(
                input_column='title',
                output_column='psa_grade',
                instruction="""Extract only the PSA grade number from this text.
                             Return only the number (e.g., '9.8', '7.0').
                             If no PSA grade is mentioned, return 'None'.
                             Do not include 'PSA' or any other text."""
            )
        ]
        
        # Initialize and run extraction
        extractor = InformationExtractor(debug=True)
        df_processed = extractor.process_dataframe_multiple(
            df=df,
            extraction_params_list=extraction_params
        )
        
        # Post-processing
        processor = DataFrameProcessor()
        df_processed = processor.normalize_case(df_processed)
        df_processed[['character', 'character2']] = df_processed['character'].str.split(r'\n', n=1, expand=True)
        df_processed = processor.delete_row_on_length(df_processed, 'character', 25)
        df_processed = processor.extract_price_shipping(df_processed, 'shipping')
        
        # First convert string 'None' to actual None/NaN values
        df_processed['psa_grade'] = df_processed['psa_grade'].replace('None', pd.NA)

        # Then drop the rows with empty values
        df_processed = df_processed.dropna(subset=['psa_grade'])

        # If you still want to remove empty strings
        df_processed = df_processed[df_processed['psa_grade'] != '']
                
        
        
        #convert sold_date to datetime
        df_processed = processor.extract_sold_date(df_processed, 'sold_date')
        #Now 'scrape_date'
        
        #Ensure datatypes are correct
        
        float_rows = ['price', 'shipping', 'psa_grade']
        str_rows = ['character', 'character2', 'condition']
        
        for row in float_rows:
            df_processed[row] = pd.to_numeric(df_processed[row], errors='coerce')
        
        for row in str_rows:
            df_processed[row] = df_processed[row].astype(str)
            
        
        df_processed = df_processed.drop(columns=['img_link', 'item_page', 'title', 'sold_date'])
        #rename date to sold_date
        df_processed.rename(columns={'date':'sold_date'}, inplace=True)
        
        
        # Save results
        output_dir = Path('processed')
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_processed.to_csv(output_file, index=False)
        logging.info(f"Saved processed data to {output_file}")
        
    except Exception as e:
        logging.error(f"Critical error in main script: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()