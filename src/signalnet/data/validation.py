"""
Data validation components for SignalNet.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class ValidationResult:
    """Result of data validation containing validation status and messages."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataValidationError(Exception):
    """Exception raised when critical data validation fails."""
    pass


class DataValidator:
    """Validates signal data for common issues and requirements."""
    
    def __init__(self, required_columns: List[str] = None):
        """
        Initialize validator with required columns.
        
        Args:
            required_columns: List of column names that must be present.
                            Defaults to ['series_id', 'timestamp', 'value']
        """
        self.required_columns = required_columns or ['series_id', 'timestamp', 'value']
    
    def validate_basic(self, df: pd.DataFrame) -> ValidationResult:
        """
        Perform basic validation checks on the dataframe.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings = []
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Only proceed with further checks if required columns exist
        if not missing_columns:
            # Check for duplicate timestamps per series
            if 'series_id' in df.columns and 'timestamp' in df.columns:
                duplicate_timestamps = self._check_duplicate_timestamps(df)
                if duplicate_timestamps:
                    errors.extend(duplicate_timestamps)
            
            # Check that values are numeric
            if 'value' in df.columns:
                numeric_errors = self._check_numeric_values(df)
                if numeric_errors:
                    errors.extend(numeric_errors)
        
        # Check for empty dataframe
        if df.empty:
            errors.append("DataFrame is empty")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> List[str]:
        """Check for duplicate timestamps within each series."""
        errors = []
        
        # Group by series_id and check for duplicate timestamps
        for series_id, group in df.groupby('series_id'):
            duplicate_count = group['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Series '{series_id}' has {duplicate_count} duplicate timestamps")
        
        return errors
    
    def _check_numeric_values(self, df: pd.DataFrame) -> List[str]:
        """Check that value column contains numeric data."""
        errors = []
        
        try:
            # Try to convert to numeric, errors='coerce' will turn non-numeric to NaN
            numeric_values = pd.to_numeric(df['value'], errors='coerce')
            non_numeric_count = numeric_values.isna().sum() - df['value'].isna().sum()
            
            if non_numeric_count > 0:
                errors.append(f"Found {non_numeric_count} non-numeric values in 'value' column")
                
        except Exception as e:
            errors.append(f"Error checking numeric values: {str(e)}")
        
        return errors