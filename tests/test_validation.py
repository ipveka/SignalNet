"""
Unit tests for data validation.
"""
import pandas as pd
import numpy as np
import pytest
from signalnet.data.validation import DataValidator, ValidationResult, DataValidationError


class TestValidationResult:
    """Test ValidationResult data class."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult with different parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        
        result = ValidationResult(is_valid=False, errors=["Error 1"], warnings=["Warning 1"])
        assert result.is_valid is False
        assert result.errors == ["Error 1"]
        assert result.warnings == ["Warning 1"]


class TestDataValidationError:
    """Test DataValidationError exception."""
    
    def test_data_validation_error(self):
        """Test raising DataValidationError."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("Test error message")


class TestDataValidator:
    """Test DataValidator class."""
    
    def test_validator_initialization_default(self):
        """Test validator initialization with default columns."""
        validator = DataValidator()
        assert validator.required_columns == ['series_id', 'timestamp', 'value']
    
    def test_validator_initialization_custom(self):
        """Test validator initialization with custom columns."""
        custom_columns = ['id', 'time', 'signal']
        validator = DataValidator(required_columns=custom_columns)
        assert validator.required_columns == custom_columns
    
    def test_validate_basic_valid_data(self):
        """Test validation with valid data."""
        df = pd.DataFrame({
            'series_id': ['A', 'A', 'B', 'B'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00', 
                                       '2023-01-01 00:00', '2023-01-01 01:00']),
            'value': [1.0, 2.0, 3.0, 4.0]
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_validate_basic_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'series_id': ['A', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00'])
            # Missing 'value' column
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Missing required columns: ['value']" in result.errors[0]
    
    def test_validate_basic_duplicate_timestamps(self):
        """Test validation with duplicate timestamps in same series."""
        df = pd.DataFrame({
            'series_id': ['A', 'A', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:00', '2023-01-01 01:00']),
            'value': [1.0, 2.0, 3.0]
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Series 'A' has 1 duplicate timestamps" in result.errors[0]
    
    def test_validate_basic_non_numeric_values(self):
        """Test validation with non-numeric values."""
        df = pd.DataFrame({
            'series_id': ['A', 'A', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00']),
            'value': [1.0, 'invalid', 3.0]
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Found 1 non-numeric values in 'value' column" in result.errors[0]
    
    def test_validate_basic_empty_dataframe(self):
        """Test validation with empty dataframe."""
        df = pd.DataFrame()
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert "DataFrame is empty" in result.errors
    
    def test_validate_basic_multiple_errors(self):
        """Test validation with multiple errors."""
        df = pd.DataFrame({
            'series_id': ['A', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:00']),
            'value': [1.0, 'invalid']
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        # Should have both duplicate timestamp and non-numeric value errors
        error_text = ' '.join(result.errors)
        assert "duplicate timestamps" in error_text
        assert "non-numeric values" in error_text
    
    def test_validate_basic_missing_series_id_column(self):
        """Test validation when series_id column is missing."""
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00']),
            'value': [1.0, 2.0]
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert "Missing required columns: ['series_id']" in result.errors[0]
    
    def test_validate_basic_nan_values_allowed(self):
        """Test that NaN values in value column are allowed (not treated as non-numeric)."""
        df = pd.DataFrame({
            'series_id': ['A', 'A', 'A'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00']),
            'value': [1.0, np.nan, 3.0]
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is True
        assert result.errors == []
    
    def test_check_duplicate_timestamps_multiple_series(self):
        """Test duplicate timestamp checking across multiple series."""
        df = pd.DataFrame({
            'series_id': ['A', 'A', 'B', 'B', 'B'],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:00',  # A has duplicate
                                       '2023-01-01 00:00', '2023-01-01 00:00', '2023-01-01 00:00']),  # B has 2 duplicates
            'value': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        validator = DataValidator()
        result = validator.validate_basic(df)
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        # Should have errors for both series
        error_text = ' '.join(result.errors)
        assert "Series 'A' has 1 duplicate timestamps" in error_text
        assert "Series 'B' has 2 duplicate timestamps" in error_text