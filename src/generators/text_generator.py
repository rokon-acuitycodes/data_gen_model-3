import io
from pathlib import Path
from typing import IO, List, Tuple

import pandas as pd
from docx import Document
from pypdf import PdfReader

from generators.base import DataGenerator
from models.t5 import T5Model
from utils.helpers import get_paragraphs


MAX_TABULAR_ROWS = 50
MAX_SOURCE_PARAGRAPHS = 12
TEXT_REWRITE_INSTRUCTION = (
    "Rewrite the following content as clear, natural text while preserving the key facts:"
)


class TextGenerator(DataGenerator):
    """Generate plain-text outputs from uploaded files using FLAN-T5."""

    def __init__(self, t5_model: T5Model):
        self.t5_model = t5_model

    def _extract_text(self, original_file: IO[bytes], file_ext: str) -> str:
        original_file.seek(0)

        if file_ext == "txt":
            raw_bytes = original_file.read()
            for encoding in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    return raw_bytes.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw_bytes.decode("utf-8", errors="ignore")

        if file_ext == "docx":
            document = Document(original_file)
            return "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())

        if file_ext == "pdf":
            reader = PdfReader(original_file)
            page_text = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(text.strip() for text in page_text if text.strip())

        if file_ext == "csv":
            dataframe = pd.read_csv(original_file).fillna("")
            return self._tabular_to_text(dataframe)

        if file_ext == "xlsx":
            dataframe = pd.read_excel(original_file).fillna("")
            return self._tabular_to_text(dataframe)

        raise ValueError(f"Unsupported file format for text generation: {file_ext}")

    def _tabular_to_text(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "The uploaded table is empty."

        sample = dataframe.head(MAX_TABULAR_ROWS).astype(str)
        row_text = []
        for row_idx, row in sample.iterrows():
            values = [f"{column}: {value}" for column, value in row.items() if str(value).strip()]
            if values:
                row_text.append(f"Row {row_idx + 1}: " + ", ".join(values))

        columns = ", ".join(sample.columns.astype(str).tolist())
        lines = [f"Columns: {columns}"]
        lines.extend(row_text)
        return "\n".join(lines)

    def _extract_paragraphs(self, original_file: IO[bytes], file_ext: str) -> List[str]:
        text = self._extract_text(original_file, file_ext)
        paragraphs = get_paragraphs(text)
        if not paragraphs:
            cleaned_text = text.strip()
            return [cleaned_text] if cleaned_text else []
        return paragraphs[:MAX_SOURCE_PARAGRAPHS]

    def generate_texts(
        self,
        original_file: IO[bytes],
        file_ext: str,
        num_outputs: int = 3,
        max_length: int = 200,
    ) -> List[str]:
        paragraphs = self._extract_paragraphs(original_file, file_ext)
        if not paragraphs:
            return []

        generated_outputs = []
        for _ in range(num_outputs):
            generated_paragraphs = self.t5_model.generate_batch_from_instruction(
                input_texts=paragraphs,
                instruction=TEXT_REWRITE_INSTRUCTION,
                max_length=max_length,
                batch_size=4,
            )
            generated_text = "\n\n".join(
                paragraph.strip() for paragraph in generated_paragraphs if paragraph.strip()
            ).strip()
            if generated_text:
                generated_outputs.append(generated_text)

        return generated_outputs

    def generate(
        self,
        original_file: IO[bytes],
        num_files: int = 3,
        file_ext: str = "txt",
        max_length: int = 200,
        **kwargs,
    ) -> List[Tuple[str, bytes]]:
        generated_texts = self.generate_texts(
            original_file=original_file,
            file_ext=file_ext,
            num_outputs=num_files,
            max_length=max_length,
        )

        original_name = Path(getattr(original_file, "name", "upload")).stem
        return [
            (
                f"generated_text_{index + 1}_{original_name}.txt",
                text.encode("utf-8"),
            )
            for index, text in enumerate(generated_texts)
        ]
