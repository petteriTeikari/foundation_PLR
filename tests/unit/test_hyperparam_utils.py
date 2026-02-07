"""Unit tests for hyperparameter naming utilities.

Tests functions from src/orchestration/hyperparamer_list_utils.py for
hyperparameter name creation and formatting.
"""

import pytest

from src.orchestration.hyperparamer_list_utils import (
    clean_param,
    create_hyperparam_name,
)


class TestCleanParam:
    """Tests for clean_param function."""

    @pytest.mark.unit
    def test_clean_param_single_word(self):
        """Test that single word parameter is unchanged."""
        assert clean_param("epochs") == "epochs"
        assert clean_param("batch") == "batch"
        assert clean_param("lr") == "lr"

    @pytest.mark.unit
    def test_clean_param_underscore_to_camelcase(self):
        """Test conversion of snake_case to camelCase."""
        assert clean_param("d_ffn") == "dFfn"
        assert clean_param("n_layers") == "nLayers"
        assert clean_param("learning_rate") == "learningRate"

    @pytest.mark.unit
    def test_clean_param_multiple_underscores(self):
        """Test handling of multiple underscores."""
        assert clean_param("d_ffn_hidden") == "dFfnHidden"
        assert clean_param("num_attention_heads") == "numAttentionHeads"

    @pytest.mark.unit
    def test_clean_param_numeric_suffix(self):
        """Test parameters with numeric parts."""
        assert clean_param("layer_1") == "layer1"
        assert clean_param("dim_128") == "dim128"

    @pytest.mark.unit
    def test_clean_param_title_case(self):
        """Test that subsequent words get title case (lowercase then capitalize)."""
        # The function applies .title() which converts to title case
        # FFN -> Ffn (title case)
        assert clean_param("d_FFN") == "dFfn"

    @pytest.mark.unit
    def test_clean_param_empty_string(self):
        """Test empty string handling."""
        assert clean_param("") == ""


class TestCreateHyperparamName:
    """Tests for create_hyperparam_name function."""

    @pytest.mark.unit
    def test_create_name_basic(self):
        """Test basic hyperparameter name creation."""
        result = create_hyperparam_name(
            param="epochs",
            value_from_list=100,
            i=0,
            j=0,
            n_params=1,
        )
        assert result == "epochs100"

    @pytest.mark.unit
    def test_create_name_with_delimiter(self):
        """Test name creation with param delimiter (not last param)."""
        result = create_hyperparam_name(
            param="epochs",
            value_from_list=100,
            i=0,
            j=0,
            n_params=2,  # Not the last param
            param_delimiter="_",
        )
        assert result == "epochs100_"

    @pytest.mark.unit
    def test_create_name_last_param_no_delimiter(self):
        """Test that last parameter doesn't get trailing delimiter."""
        result = create_hyperparam_name(
            param="lr",
            value_from_list=0.001,
            i=0,
            j=1,  # Second param (index 1)
            n_params=2,  # Last param when j+1 == n_params
            param_delimiter="_",
        )
        assert result == "lr0.001"
        assert not result.endswith("_")

    @pytest.mark.unit
    def test_create_name_value_key_delimiter(self):
        """Test name creation with value key delimiter."""
        result = create_hyperparam_name(
            param="epochs",
            value_from_list=100,
            i=0,
            j=0,
            n_params=1,
            value_key_delimiter="=",
        )
        assert result == "epochs=100"

    @pytest.mark.unit
    def test_create_name_snake_case_param(self):
        """Test that snake_case params are converted."""
        result = create_hyperparam_name(
            param="d_ffn",
            value_from_list=256,
            i=0,
            j=0,
            n_params=1,
        )
        assert result == "dFfn256"

    @pytest.mark.unit
    def test_create_name_float_value(self):
        """Test with float values."""
        result = create_hyperparam_name(
            param="learning_rate",
            value_from_list=0.001,
            i=0,
            j=0,
            n_params=1,
        )
        assert result == "learningRate0.001"

    @pytest.mark.unit
    def test_create_name_string_value(self):
        """Test with string values."""
        result = create_hyperparam_name(
            param="activation",
            value_from_list="relu",
            i=0,
            j=0,
            n_params=1,
        )
        assert result == "activationrelu"

    @pytest.mark.unit
    def test_create_name_multiple_params_sequence(self):
        """Test creating names for a sequence of parameters."""
        params = [("epochs", 100), ("lr", 0.01), ("batch", 32)]
        n_params = len(params)

        names = []
        for j, (param, value) in enumerate(params):
            name = create_hyperparam_name(
                param=param,
                value_from_list=value,
                i=0,
                j=j,
                n_params=n_params,
                param_delimiter="_",
            )
            names.append(name)

        combined = "".join(names)
        assert combined == "epochs100_lr0.01_batch32"

    @pytest.mark.unit
    def test_create_name_custom_delimiters(self):
        """Test with custom delimiter combinations."""
        result = create_hyperparam_name(
            param="epochs",
            value_from_list=100,
            i=0,
            j=0,
            n_params=2,
            value_key_delimiter="-",
            param_delimiter="__",
        )
        assert result == "epochs-100__"
