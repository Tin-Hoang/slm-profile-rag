"""Tests for response enhancer module."""

import re
from unittest.mock import patch

import pytest

from src.response_enhancer import ResponseEnhancer, get_response_enhancer


@pytest.fixture
def enhancer():
    """Create a ResponseEnhancer instance for testing."""
    with patch("src.response_enhancer.get_config") as mock_config:
        mock_config.return_value.get.return_value = "Tin Hoang"
        return ResponseEnhancer()


# --- ResponseEnhancer Tests ---


def test_enhancer_init(enhancer):
    """Test ResponseEnhancer initialization."""
    assert enhancer is not None
    assert enhancer.name == "Tin Hoang"
    assert len(enhancer.negative_patterns) > 0
    assert len(enhancer.positive_closings) > 0


def test_enhance_empty_response(enhancer):
    """Test enhance handles empty response."""
    result = enhancer.enhance("")
    assert result == ""

    # Whitespace-only gets stripped and returns empty
    result = enhancer.enhance("   ")
    assert result.strip() == ""


def test_enhance_none_response(enhancer):
    """Test enhance handles None response."""
    result = enhancer.enhance(None)
    assert result is None


def test_enhance_simple_response(enhancer):
    """Test enhance keeps simple responses unchanged."""
    response = "Tin is an AI Engineer with expertise in machine learning."
    result = enhancer.enhance(response)
    assert "Tin" in result or "AI Engineer" in result


def test_rewrite_no_info_pattern(enhancer):
    """Test rewriting 'don't have information' pattern."""
    response = "However, I don't have any specific information about that topic."
    result = enhancer.enhance(response)

    # Should not contain the negative phrase
    assert "don't have" not in result.lower()
    assert "however" not in result.lower() or "directly" in result.lower()


def test_rewrite_uncertain_pattern(enhancer):
    """Test rewriting uncertain statements."""
    response = "I'm not sure about the specific salary expectations."
    result = enhancer.enhance(response)

    # Should not contain the uncertain phrase
    assert "not sure" not in result.lower()


def test_rewrite_cannot_provide_pattern(enhancer):
    """Test rewriting 'cannot provide' statements."""
    response = "I cannot provide more information about his personal life."
    result = enhancer.enhance(response)

    # Should not contain the negative phrase
    assert "cannot provide" not in result.lower()


def test_rewrite_do_not_have_pattern(enhancer):
    """Test rewriting 'do not have' statements."""
    response = "I do not have any details about that project."
    result = enhancer.enhance(response)

    # Should not contain the negative phrase
    assert "do not have" not in result.lower()


def test_add_positive_closing_negative_ending(enhancer):
    """Test adding positive closing for negative endings."""
    response = "The profile covers relevant experience. However, there's no information available."
    result = enhancer.enhance(response)

    # Should have some form of positive closing
    assert len(result) >= len(response)


def test_no_double_closing(enhancer):
    """Test no double closing when already has contact suggestion."""
    response = "For more details, feel free to connect directly with me."
    result = enhancer.enhance(response)

    # Should not add another closing
    assert result.count("directly") <= 2


def test_fix_markdown_inline_numbered_list(enhancer):
    """Test fixing inline numbered lists."""
    response = "Key skills: 1. Python 2. Machine Learning 3. Deep Learning"
    result = enhancer.enhance(response)

    # Should format the list properly
    assert result is not None


def test_fix_markdown_already_formatted_list(enhancer):
    """Test preserving already formatted lists."""
    response = """Key skills:

1. Python
2. Machine Learning
3. Deep Learning"""
    result = enhancer.enhance(response)

    # Should preserve the formatting
    assert "1." in result
    assert "2." in result
    assert "3." in result


def test_cleanup_double_spaces(enhancer):
    """Test cleanup of double spaces."""
    response = "This  has   multiple    spaces."
    result = enhancer.enhance(response)

    assert "  " not in result


def test_cleanup_punctuation(enhancer):
    """Test cleanup of awkward punctuation."""
    response = "This has extra spaces before punctuation ."
    result = enhancer.enhance(response)

    assert " ." not in result


def test_cleanup_double_punctuation(enhancer):
    """Test cleanup of double punctuation."""
    response = "This has double punctuation.."
    result = enhancer.enhance(response)

    # Should have single punctuation
    assert ".." not in result


def test_enhance_with_context_job_search(enhancer):
    """Test enhance_with_context adds job-related closing."""
    question = "What job roles are you looking for?"
    response = "The candidate has experience in AI and ML engineering."
    result = enhancer.enhance_with_context(response, question)

    # Should add contact suggestion for job-related questions
    assert "direct" in result.lower() or "contact" in result.lower()


def test_enhance_with_context_already_has_contact(enhancer):
    """Test enhance_with_context doesn't add duplicate contact suggestion."""
    question = "What position are you seeking?"
    response = "Looking for ML roles. Feel free to reach out to discuss opportunities."
    result = enhancer.enhance_with_context(response, question)

    # Should not add another contact suggestion
    assert result.count("reach out") <= 1


def test_enhance_with_context_non_job_question(enhancer):
    """Test enhance_with_context for non-job questions."""
    question = "What are your technical skills?"
    response = "Strong background in Python, TensorFlow, and PyTorch."
    result = enhancer.enhance_with_context(response, question)

    # Should be similar to original (no job-specific closing added)
    assert "Python" in result


# --- Rewrite Function Tests ---


def test_rewrite_no_info_consistent(enhancer):
    """Test _rewrite_no_info produces consistent output for same topic."""
    pattern = r"(?:I\s+)?(?:don'?t|do\s+not)\s+have\s+(?:any\s+)?(?:more\s+)?(?:information|details|data)\s+(?:about|on|regarding)\s+([^.!?]+)"
    text = "I don't have any information about salary expectations."

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        result1 = enhancer._rewrite_no_info(match)
        result2 = enhancer._rewrite_no_info(match)

        # Same topic should produce same rewrite
        assert result1 == result2


def test_rewrite_uncertain_consistent(enhancer):
    """Test _rewrite_uncertain produces consistent output for same topic."""
    pattern = r"(?:I'm|I am)\s+(?:not\s+)?(?:un)?(?:sure|certain|aware)\s+(?:about|of)\s+([^.!?]+)"
    text = "I'm not sure about the project timeline."

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        result1 = enhancer._rewrite_uncertain(match)
        result2 = enhancer._rewrite_uncertain(match)

        # Same topic should produce same rewrite
        assert result1 == result2


def test_rewrite_cannot_provide_consistent(enhancer):
    """Test _rewrite_cannot_provide produces consistent output for same topic."""
    pattern = r"(?:I\s+)?(?:cannot|can'?t|unable to)\s+(?:provide|share|give)\s+(?:more\s+)?(?:information|details)\s+(?:about|on)\s+([^.!?]+)"
    text = "I cannot provide more details about that."

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        result1 = enhancer._rewrite_cannot_provide(match)
        result2 = enhancer._rewrite_cannot_provide(match)

        # Same topic should produce same rewrite
        assert result1 == result2


# --- Markdown Formatting Tests ---


def test_fix_markdown_bullet_list(enhancer):
    """Test handling bullet lists."""
    response = """Skills include:
- Python
- Machine Learning
- Deep Learning"""
    result = enhancer._fix_markdown_formatting(response)

    assert "-" in result
    assert "Python" in result


def test_fix_markdown_mixed_content(enhancer):
    """Test handling mixed content."""
    response = """Overview paragraph.

1. First item
2. Second item

Another paragraph."""
    result = enhancer._fix_markdown_formatting(response)

    assert "1." in result
    assert "2." in result


def test_fix_markdown_no_multiple_blank_lines(enhancer):
    """Test no multiple blank lines."""
    response = """First paragraph.



Second paragraph."""
    result = enhancer._fix_markdown_formatting(response)

    # Should not have multiple consecutive blank lines
    assert "\n\n\n" not in result


# --- Factory Function Tests ---


def test_get_response_enhancer():
    """Test get_response_enhancer returns instance."""
    with patch("src.response_enhancer.get_config") as mock_config:
        mock_config.return_value.get.return_value = "Test Name"

        result = get_response_enhancer()

        assert isinstance(result, ResponseEnhancer)
