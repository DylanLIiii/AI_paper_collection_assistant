from datetime import datetime
from typing import List, Optional, Dict, Any
from loguru import logger
import os
import arxiv
from markitdown import MarkItDown


class BasePaper:
    """Base class for all paper types with common functionality."""
    
    REQUIRED_ATTRIBUTES = {
        'id': None,  # or arxiv_id
        'title': None,
        'abstract': None,
        'authors': [],
        'url': None,
    }
    
    def __init__(self, **kwargs):
        """
        Initialize a paper with dynamic attributes.
        Common attributes include: id/arxiv_id, title, abstract, authors, url
        """
        # Private attribute for lazy loading of PDF content
        self._pdf_content = None
        
        # Dynamically assign all attributes
        self._set_attributes(kwargs)
        
        # Validate and ensure basic attributes exist
        self._validate_attributes()

    def _set_attributes(self, attrs: Dict[str, Any]) -> None:
        """Set attributes dynamically from input dictionary."""
        for key, value in attrs.items():
            setattr(self, key, value)

    def _validate_attributes(self) -> None:
        """Ensure all required attributes exist with proper defaults."""
        # Handle ID (can be either 'id' or 'arxiv_id')
        self.id = getattr(self, 'id', None) or getattr(self, 'arxiv_id', None)
        if hasattr(self, 'arxiv_id'):
            self.id = self.arxiv_id
        
        # Ensure other required attributes exist
        for attr, default in self.REQUIRED_ATTRIBUTES.items():
            if not hasattr(self, attr) or getattr(self, attr) is None:
                setattr(self, attr, default)
                logger.debug(f"Setting default value for {attr}: {default}")

    @property
    def pdf_content(self) -> Optional[str]:
        """
        Lazy load and cache the PDF content.
        Returns None if PDF cannot be loaded.
        """
        if self._pdf_content is None:
            self._pdf_content = self._load_pdf_content()
        return self._pdf_content

    def _load_pdf_content(self) -> Optional[str]:
        """
        Load PDF content using arxiv API and markitdown.
        Returns None if loading fails.
        """
        try:
            # Ensure we have an arxiv ID
            if not hasattr(self, 'arxiv_id') and not hasattr(self, 'id'):
                logger.warning("No arxiv_id available to load PDF content")
                return None

            arxiv_id = getattr(self, 'arxiv_id', None) or self.id
            
            # Create output directory if it doesn't exist
            # TODO: We should use a better directory to store the PDFs. As temporary solution, we use the output directory
            pdf_dir = "out/pdfs"
            os.makedirs(pdf_dir, exist_ok=True)
            
            pdf_path = f"{pdf_dir}/{arxiv_id}.pdf"
            
            # Download PDF if it doesn't exist
            if not os.path.exists(pdf_path):
                search = arxiv.Search(id_list=[arxiv_id])
                results = list(arxiv.Client().results(search))
                if not results:
                    logger.warning(f"No results found for arxiv ID {arxiv_id}")
                    return None
                    
                paper_entry = next(results)
                paper_entry.download_pdf(filename=pdf_path)
            
            # Convert PDF to text
            # TODO: We use MarkItDonw as the default PDF to text converter, but it is not very good, we should use a better one
            md = MarkItDown()
            result = md.convert(pdf_path)
            return result.text_content

        except Exception as e:
            logger.error(f"Error loading PDF content for paper {self.id}: {e}")
            return None

    def __repr__(self) -> str:
        """Return a readable string representation of the object."""
        attrs = {
            key: getattr(self, key, None)
            for key in ["id", "title", "authors"]
            if hasattr(self, key)
        }
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in attrs.items())})"

    def __hash__(self) -> int:
        """Hash based on paper ID."""
        return hash(self.id)

    def __eq__(self, other) -> bool:
        """Equality based on paper ID."""
        if not isinstance(other, BasePaper):
            return False
        return self.id == other.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert the Paper object to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            # Skip private attributes and PDF content
            if key.startswith('_'):
                continue
            if isinstance(value, datetime):
                result[key] = value.isoformat() if value else None
            else:
                result[key] = value
        return result 