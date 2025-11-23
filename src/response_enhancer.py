"""Response enhancement module for improving RAG outputs."""

import logging
import re

from .config_loader import get_config

logger = logging.getLogger(__name__)


class ResponseEnhancer:
    """Enhance and polish RAG responses for better user experience."""

    def __init__(self):
        """Initialize response enhancer with configuration."""
        self.config = get_config()
        self.name = self.config.get("profile.name", "the candidate")

        # Negative phrases to detect and rewrite
        self.negative_patterns = [
            (
                r"(?:However|Unfortunately|I apologize),?\s*(?:I\s+)?don'?t\s+have\s+(?:any\s+)?(?:more\s+)?(?:specific\s+)?(?:information|details)\s+(?:about|on|regarding)\s+([^.!?]+)",
                self._rewrite_no_info,
            ),
            (
                r"(?:I\s+)?(?:don'?t|do\s+not)\s+have\s+(?:any\s+)?(?:more\s+)?(?:information|details|data)\s+(?:about|on|regarding)\s+([^.!?]+)",
                self._rewrite_no_info,
            ),
            (
                r"(?:I'm|I am)\s+(?:not\s+)?(?:un)?(?:sure|certain|aware)\s+(?:about|of)\s+([^.!?]+)",
                self._rewrite_uncertain,
            ),
            (
                r"(?:I\s+)?(?:cannot|can'?t|unable to)\s+(?:provide|share|give)\s+(?:more\s+)?(?:information|details)\s+(?:about|on)\s+([^.!?]+)",
                self._rewrite_cannot_provide,
            ),
        ]

        # Closing enhancements (more modest)
        self.positive_closings = [
            "For more details, connecting directly would be helpful.",
            "Additional information can be discussed in a direct conversation.",
            "Feel free to reach out for more specific information.",
            "Direct contact would provide more comprehensive details.",
        ]

    def _rewrite_no_info(self, match: re.Match) -> str:
        """Rewrite 'don't have information' statements.

        Args:
            match: Regex match object

        Returns:
            Rewritten positive statement
        """
        topic = match.group(1).strip()
        rewrites = [
            f"For more specific details about {topic}, it would be best to connect directly.",
            f"The available profile focuses on other aspects of the background.",
            f"Additional information about {topic} can be discussed in a direct conversation.",
            f"The documented profile covers the key highlights.",
        ]
        # Use hash to consistently select same rewrite for same topic
        idx = hash(topic) % len(rewrites)
        return rewrites[idx]

    def _rewrite_uncertain(self, match: re.Match) -> str:
        """Rewrite uncertain statements.

        Args:
            match: Regex match object

        Returns:
            Rewritten confident statement
        """
        topic = match.group(1).strip()
        rewrites = [
            f"Specific details about {topic} would be best discussed directly.",
            f"For more information about {topic}, connecting directly would be helpful.",
            f"Details about {topic} may be available through direct conversation.",
        ]
        idx = hash(topic) % len(rewrites)
        return rewrites[idx]

    def _rewrite_cannot_provide(self, match: re.Match) -> str:
        """Rewrite 'cannot provide' statements.

        Args:
            match: Regex match object

        Returns:
            Rewritten forward-looking statement
        """
        topic = match.group(1).strip()
        rewrites = [
            f"More detailed information about {topic} can be discussed directly.",
            f"For additional context about {topic}, direct conversation would be helpful.",
            f"Further details about {topic} are available through direct contact.",
        ]
        idx = hash(topic) % len(rewrites)
        return rewrites[idx]

    def _add_positive_closing(self, text: str) -> str:
        """Add a modest closing statement if appropriate.

        Args:
            text: Response text

        Returns:
            Text with closing added if needed (only for very negative endings)
        """
        # Check if response already mentions connecting or reaching out
        forward_looking_patterns = [
            r"(?:connect|contact|reach out|discuss|conversation|directly)",
            r"(?:feel free|available|happy|welcome)\s+to",
        ]

        for pattern in forward_looking_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return text  # Already has some form of closing or contact suggestion

        # Only add closing for clearly negative endings
        very_negative_ending_patterns = [
            r"(?:however|unfortunately).*(?:don'?t|cannot|can'?t|no)[^.!?]*[.!?]\s*$",
            r"(?:not\s+available|not\s+found|no\s+information)[^.!?]*[.!?]\s*$",
        ]

        for pattern in very_negative_ending_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Add a modest closing
                closing = self.positive_closings[hash(text) % len(self.positive_closings)]
                return f"{text.rstrip()} {closing}"  # Single space, not double newline

        return text

    def _fix_markdown_formatting(self, text: str) -> str:
        """Fix markdown formatting issues, particularly inline numbered lists.

        Args:
            text: Response text with potential formatting issues

        Returns:
            Text with properly formatted markdown
        """
        # Simple but effective approach: Replace inline list patterns with line-broken versions

        # Step 1: Fix inline numbered lists
        # Pattern: Find any text that has "1. text 2. text" (not already on separate lines)
        # We'll use a simple replace: put newline before each " N. " where N is a digit

        # But first, protect already well-formatted lists (those at start of line)
        lines = text.split('\n')
        protected_lines = []

        for line in lines:
            stripped = line.strip()
            # Check if line starts with a number (already formatted list)
            if re.match(r'^\d+\.\s+', stripped):
                # Protect by temporarily marking it
                protected_lines.append('___LISTITEM___' + line)
            else:
                protected_lines.append(line)

        text = '\n'.join(protected_lines)

        # Now fix inline lists: add line break before each " 1.", " 2.", etc.
        # But only if it's in the middle of text (has non-whitespace before it)
        text = re.sub(r'([^\n])\s+(\d+)\.\s+([A-Z])', r'\1\n\n\2. \3', text)

        # Remove protection markers
        text = text.replace('___LISTITEM___', '')

        # Step 2: Ensure proper blank lines between elements
        lines = text.split('\n')
        formatted_lines = []
        prev_was_list = False

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Don't add multiple blank lines in a row
                if formatted_lines and formatted_lines[-1] != '':
                    formatted_lines.append('')
                continue

            is_list = bool(re.match(r'^\d+\.|^-\s+|^\*\s+', stripped))

            # Add blank line transitions
            if formatted_lines:
                last_line = formatted_lines[-1]

                # Add blank before first list item
                if is_list and not prev_was_list and last_line != '':
                    formatted_lines.append('')

                # Add blank after last list item (before regular paragraph)
                elif not is_list and prev_was_list and last_line != '':
                    formatted_lines.append('')

            formatted_lines.append(line)
            prev_was_list = is_list

        return '\n'.join(formatted_lines)

    def enhance(self, response: str) -> str:
        """Enhance response by removing negative language and adding positive tone.

        Args:
            response: Original response text

        Returns:
            Enhanced response text
        """
        if not response or len(response.strip()) == 0:
            return response

        enhanced = response

        # Apply negative pattern rewrites
        for pattern, rewrite_func in self.negative_patterns:
            matches = list(re.finditer(pattern, enhanced, re.IGNORECASE))
            # Process matches in reverse to maintain string positions
            for match in reversed(matches):
                replacement = rewrite_func(match)
                enhanced = enhanced[: match.start()] + replacement + enhanced[match.end() :]
                logger.debug(f"Rewrote negative phrase: '{match.group(0)}' -> '{replacement}'")

        # Add positive closing if needed
        enhanced = self._add_positive_closing(enhanced)

        # Fix markdown formatting
        enhanced = self._fix_markdown_formatting(enhanced)

        # Clean up any double spaces or awkward punctuation
        enhanced = re.sub(r" +", " ", enhanced)  # Multiple spaces to single space
        enhanced = re.sub(r"\s+([.,!?])", r"\1", enhanced)
        enhanced = re.sub(r"([.!?])\s*([.!?])", r"\1", enhanced)

        return enhanced.strip()

    def enhance_with_context(self, response: str, question: str) -> str:
        """Enhance response with awareness of the question context.

        Args:
            response: Original response text
            question: Original question

        Returns:
            Context-aware enhanced response
        """
        enhanced = self.enhance(response)

        # If the question is about job search/roles and response seems incomplete
        # Add a modest closing about opportunities (not overly enthusiastic)
        if re.search(r"(?:job|role|position|opportunity|looking for|seeking)", question, re.IGNORECASE):
            if not re.search(
                r"(?:opportunity|interview|connect|discuss|reach out|contact)", enhanced, re.IGNORECASE
            ):
                enhanced += " For current opportunities and detailed discussions, direct contact would be best."

        return enhanced


def get_response_enhancer() -> ResponseEnhancer:
    """Get response enhancer instance.

    Returns:
        ResponseEnhancer instance
    """
    return ResponseEnhancer()

