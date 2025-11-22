"""Hugging Face Spaces entry point."""

from slm_profile_rag.app import launch_ui

if __name__ == "__main__":
    # For Hugging Face Spaces deployment - use 0.0.0.0 to accept external connections
    launch_ui(share=False, server_name="0.0.0.0")
