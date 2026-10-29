import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ingestion.webscraping_utils import (
    build_web_documents,
    extract_text_from_html,
    load_urls_from_file,
)


class WebScrapingIngestionTests(unittest.TestCase):
    def test_extract_text_from_html_strips_tags_and_keeps_content(self):
        html = "<html><body><h1>Product Name</h1><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        text = extract_text_from_html(html, "https://example.com/product")
        self.assertIn("Product Name", text)
        self.assertIn("First paragraph.", text)
        self.assertIn("Second paragraph.", text)

    def test_load_urls_from_file_reads_lines_without_comments(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "urls.txt"
            path.write_text(
                "https://example.com/a\n# comment\nhttps://example.com/b\n", encoding="utf-8"
            )
            urls = load_urls_from_file(path)
            self.assertEqual(urls, ["https://example.com/a", "https://example.com/b"])

    def test_build_web_documents_uses_scraped_content(self):
        with patch("ingestion.webscraping_utils.requests.get") as mock_get:
            mock_get.return_value.raise_for_status.return_value = None
            mock_get.return_value.text = "<html><head><title>Demo Product</title></head><body><h1>Demo Product</h1><p>Useful details.</p></body></html>"

            documents = build_web_documents(["https://example.com/demo"])

            self.assertEqual(len(documents), 1)
            self.assertIn("Useful details.", documents[0].page_content)
            self.assertEqual(documents[0].metadata["source"], "https://example.com/demo")


if __name__ == "__main__":
    unittest.main()
