from paper_assistant.cli.commands import main
import os

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("out", exist_ok=True)
    os.makedirs("out/cache", exist_ok=True)
    
    # Run the CLI
    main()
