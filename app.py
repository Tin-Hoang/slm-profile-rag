"""Hugging Face Spaces entry point."""

from slm_profile_rag.app import launch_ui

if __name__ == "__main__":
    # For Hugging Face Spaces deployment
    launch_ui(share=False)
